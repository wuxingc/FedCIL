"""
Boundary Erosion Score (BES) 诊断工具
用于计算和记录旧类边界侵蚀风险，验证BES能否预测forgetting
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import os
import csv
from collections import defaultdict


class BESDiagnostics:
    """BES诊断类，用于计算边界侵蚀分数并记录数据"""

    def __init__(self, learner, output_dir="run/bes_diagnostics"):
        self.learner = learner
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 存储每个任务的数据
        self.task_data = defaultdict(dict)  # {task_id: {class_id: {...}}}
        self.prev_task_acc = {}  # 上一任务结束时的per-class accuracy

        self.csv_path = os.path.join(output_dir, "class_diagnostics.csv")
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'class_id', 'replay_count',
                'margin_drop', 'new_intrusion', 'prototype_drift',
                'client_disagreement', 'effective_support', 'BES',
                'acc_before', 'acc_after', 'acc_drop'
            ])

    def compute_bes_for_task(self, task_id):
        """
        在任务结束时计算所有旧类的BES

        Args:
            task_id: 当前任务ID
        """
        if task_id == 0:
            # 第一个任务没有旧类
            return

        num_old_classes = self.learner._known_classes
        print(f"\n=== Computing BES for Task {task_id}, Old Classes: 0-{num_old_classes-1} ===")

        # 1. 计算每个旧类的BES组件
        for class_id in range(num_old_classes):
            bes_components = self._compute_class_bes(class_id, task_id)

            # 2. 计算replay count
            replay_count = self._count_replay_samples(class_id)

            # 3. 记录当前accuracy（作为下一任务的acc_before）
            current_acc = self._compute_class_accuracy(class_id)

            # 4. 存储数据
            self.task_data[task_id][class_id] = {
                'replay_count': replay_count,
                'margin_drop': bes_components['margin_drop'],
                'new_intrusion': bes_components['new_intrusion'],
                'prototype_drift': bes_components['prototype_drift'],
                'client_disagreement': bes_components['client_disagreement'],
                'effective_support': bes_components['effective_support'],
                'BES': bes_components['BES'],
                'acc_before': current_acc,
                'acc_after': None,  # 下一任务后填充
                'acc_drop': None
            }

            print(f"Class {class_id}: BES={bes_components['BES']:.4f}, "
                  f"Replay={replay_count}, Acc={current_acc:.2f}%")

        # 5. 如果不是第一次有旧类，更新上一任务的acc_after和acc_drop
        if task_id > 1:
            self._update_previous_task_forgetting(task_id - 1)

        # 6. 保存当前任务数据到CSV
        self._save_to_csv(task_id)

    def _compute_class_bes(self, class_id, task_id):
        """计算单个类的BES各组件"""

        # 收集该类的所有replay样本
        class_samples = self._collect_class_samples(class_id)

        if len(class_samples) == 0:
            return {
                'margin_drop': 0.0,
                'new_intrusion': 0.0,
                'prototype_drift': 0.0,
                'client_disagreement': 0.0,
                'effective_support': 1.0,
                'BES': 0.0
            }

        # 1. Margin Drop
        margin_drop = self._compute_margin_drop(class_samples, class_id)

        # 2. New-Class Intrusion
        new_intrusion = self._compute_new_intrusion(class_samples, class_id)

        # 3. Prototype Drift
        prototype_drift = self._compute_prototype_drift(class_samples, class_id)

        # 4. Client Disagreement
        client_disagreement = self._compute_client_disagreement(class_id)

        # 5. Effective Support
        effective_support = 1.0 / np.sqrt(len(class_samples) + 1e-8)

        # 6. 组合成BES (可调整权重)
        BES = (0.3 * margin_drop +
               0.3 * new_intrusion +
               0.2 * prototype_drift +
               0.1 * client_disagreement +
               0.1 * effective_support)

        return {
            'margin_drop': margin_drop,
            'new_intrusion': new_intrusion,
            'prototype_drift': prototype_drift,
            'client_disagreement': client_disagreement,
            'effective_support': effective_support,
            'BES': BES
        }

    def _collect_class_samples(self, class_id):
        """收集所有客户端中该类的replay样本"""
        samples = []

        for client_idx in range(self.learner.args['num_users']):
            if not self.learner.retained_ds_all[client_idx]:
                continue

            # 合并该客户端的所有retained datasets
            client_mem = ConcatDataset(self.learner.retained_ds_all[client_idx])

            # 提取该类的样本
            for i in range(len(client_mem)):
                try:
                    _, img, label = client_mem[i]
                    if label == class_id:
                        samples.append((img, label))
                except:
                    continue

        return samples

    def _compute_margin_drop(self, samples, class_id):
        """计算margin下降"""
        if not hasattr(self.learner, '_old_network') or self.learner._old_network is None:
            return 0.0

        self.learner._network.eval()
        self.learner._old_network.eval()

        margin_drops = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:  # 最多采样100个
                img = img.unsqueeze(0).cuda()

                # 旧模型的margin
                old_logits = self.learner._old_network(img)['logits'][0]
                old_margin = old_logits[class_id] - torch.max(
                    torch.cat([old_logits[:class_id], old_logits[class_id+1:]])
                )

                # 新模型的margin
                new_logits = self.learner._network(img)['logits'][0]
                new_margin = new_logits[class_id] - torch.max(
                    torch.cat([new_logits[:class_id], new_logits[class_id+1:]])
                )

                # margin下降（正值表示下降）
                drop = torch.clamp(old_margin - new_margin, min=0.0)
                margin_drops.append(drop.item())

        return np.mean(margin_drops) if margin_drops else 0.0

    def _compute_new_intrusion(self, samples, class_id):
        """计算新类侵入程度"""
        num_new_classes = self.learner._total_classes - self.learner._known_classes
        if num_new_classes == 0:
            return 0.0

        self.learner._network.eval()
        intrusions = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:
                img = img.unsqueeze(0).cuda()
                logits = self.learner._network(img)['logits'][0]

                # 新类的最大logit
                new_class_logits = logits[self.learner._known_classes:]
                max_new_logit = torch.max(new_class_logits)

                # 旧类的真实logit
                old_class_logit = logits[class_id]

                # 侵入程度（正值表示新类logit更高）
                intrusion = torch.clamp(max_new_logit - old_class_logit, min=0.0)
                intrusions.append(intrusion.item())

        return np.mean(intrusions) if intrusions else 0.0

    def _compute_prototype_drift(self, samples, class_id):
        """计算原型漂移"""
        if not hasattr(self.learner, '_old_network') or self.learner._old_network is None:
            return 0.0

        self.learner._network.eval()
        self.learner._old_network.eval()

        old_features = []
        new_features = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:
                img = img.unsqueeze(0).cuda()

                old_feat = self.learner._old_network(img)['features'][0]
                new_feat = self.learner._network(img)['features'][0]

                old_features.append(old_feat.cpu().numpy())
                new_features.append(new_feat.cpu().numpy())

        if not old_features:
            return 0.0

        old_proto = np.mean(old_features, axis=0)
        new_proto = np.mean(new_features, axis=0)
        old_std = np.std(old_features, axis=0).mean()

        drift = np.linalg.norm(new_proto - old_proto) / (old_std + 1e-8)
        return float(drift)

    def _compute_client_disagreement(self, class_id):
        """计算客户端间的原型差异"""
        client_protos = []

        self.learner._network.eval()

        for client_idx in range(self.learner.args['num_users']):
            if not self.learner.retained_ds_all[client_idx]:
                continue

            client_mem = ConcatDataset(self.learner.retained_ds_all[client_idx])
            class_features = []

            with torch.no_grad():
                for i in range(min(len(client_mem), 50)):
                    try:
                        _, img, label = client_mem[i]
                        if label == class_id:
                            img = img.unsqueeze(0).cuda()
                            feat = self.learner._network(img)['features'][0]
                            class_features.append(feat.cpu().numpy())
                    except:
                        continue

            if class_features:
                proto = np.mean(class_features, axis=0)
                client_protos.append(proto)

        if len(client_protos) < 2:
            return 0.0

        # 计算原型间的方差
        disagreement = np.var(client_protos, axis=0).mean()
        return float(disagreement)

    def _count_replay_samples(self, class_id):
        """统计该类的replay样本数"""
        count = 0
        for client_idx in range(self.learner.args['num_users']):
            if not self.learner.retained_ds_all[client_idx]:
                continue

            client_mem = ConcatDataset(self.learner.retained_ds_all[client_idx])
            for i in range(len(client_mem)):
                try:
                    _, _, label = client_mem[i]
                    if label == class_id:
                        count += 1
                except:
                    continue

        return count

    def _compute_class_accuracy(self, class_id):
        """计算单个类的accuracy"""
        self.learner._network.eval()

        # 获取该类的测试数据
        test_dataset = self.learner.data_manager.get_dataset(
            np.array([class_id]), source="test", mode="test"
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for _, images, labels in test_loader:
                images = images.cuda()
                outputs = self.learner._network(images)['logits']
                preds = torch.argmax(outputs, dim=1)
                correct += (preds.cpu() == labels).sum().item()
                total += len(labels)

        return 100.0 * correct / total if total > 0 else 0.0

    def _update_previous_task_forgetting(self, prev_task_id):
        """更新上一任务的forgetting数据"""
        if prev_task_id not in self.task_data:
            return

        for class_id, data in self.task_data[prev_task_id].items():
            if data['acc_after'] is None:
                # 计算当前的accuracy作为acc_after
                current_acc = self._compute_class_accuracy(class_id)
                data['acc_after'] = current_acc
                data['acc_drop'] = data['acc_before'] - current_acc

                print(f"  Class {class_id} forgetting: "
                      f"{data['acc_before']:.2f}% -> {current_acc:.2f}% "
                      f"(drop: {data['acc_drop']:.2f}%)")

    def _save_to_csv(self, task_id):
        """保存当前任务数据到CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_id, data in self.task_data[task_id].items():
                writer.writerow([
                    task_id, class_id, data['replay_count'],
                    data['margin_drop'], data['new_intrusion'],
                    data['prototype_drift'], data['client_disagreement'],
                    data['effective_support'], data['BES'],
                    data['acc_before'],
                    data['acc_after'] if data['acc_after'] is not None else '',
                    data['acc_drop'] if data['acc_drop'] is not None else ''
                ])

        print(f"Saved diagnostics to {self.csv_path}")
