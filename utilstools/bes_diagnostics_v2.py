"""
改进版 BES 诊断工具 (v2)
基于以下改进：
1. 每个任务内对各组件做归一化
2. Prototype Drift 改为 toward-new directional drift
3. Margin Drop 使用低分位数（20% quantile）
4. 加入当前脆弱性指标（current margin risk）
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import os
import csv
from collections import defaultdict


class BESDiagnosticsV2:
    """改进版BES诊断类"""

    def __init__(self, learner, output_dir="run/bes_diagnostics_v2"):
        self.learner = learner
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.task_data = defaultdict(dict)
        self.csv_path = os.path.join(output_dir, "class_diagnostics_v2.csv")
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'class_id', 'replay_count',
                'low_tail_margin_drop', 'current_margin_risk', 'new_intrusion',
                'toward_new_drift', 'client_disagreement', 'effective_support',
                'BES_v2', 'acc_before', 'acc_after', 'acc_drop',
                # 原始未归一化的值（用于调试）
                'raw_margin_drop', 'raw_margin_risk', 'raw_intrusion',
                'raw_drift', 'raw_disagreement', 'raw_support'
            ])

    def compute_bes_for_task(self, task_id):
        """在任务结束时计算所有旧类的BES v2"""
        if task_id == 0:
            return

        num_old_classes = self.learner._known_classes
        print(f"\n=== Computing BES v2 for Task {task_id}, Old Classes: 0-{num_old_classes-1} ===")

        # 先计算所有类的原始组件值
        raw_components = {}
        for class_id in range(num_old_classes):
            raw_components[class_id] = self._compute_raw_components(class_id, task_id)

        # 提取各组件的所有值用于归一化
        all_margin_drops = [c['low_tail_margin_drop'] for c in raw_components.values()]
        all_margin_risks = [c['current_margin_risk'] for c in raw_components.values()]
        all_intrusions = [c['new_intrusion'] for c in raw_components.values()]
        all_drifts = [c['toward_new_drift'] for c in raw_components.values()]
        all_disagreements = [c['client_disagreement'] for c in raw_components.values()]
        all_supports = [c['effective_support'] for c in raw_components.values()]

        # 对每个组件做min-max归一化
        def normalize(values):
            values = np.array(values)
            min_val, max_val = values.min(), values.max()
            if max_val - min_val < 1e-8:
                return np.zeros_like(values)
            return (values - min_val) / (max_val - min_val + 1e-8)

        norm_margin_drops = normalize(all_margin_drops)
        norm_margin_risks = normalize(all_margin_risks)
        norm_intrusions = normalize(all_intrusions)
        norm_drifts = normalize(all_drifts)
        norm_disagreements = normalize(all_disagreements)
        norm_supports = normalize(all_supports)

        # 计算归一化后的BES并保存
        for idx, class_id in enumerate(range(num_old_classes)):
            # 归一化后的BES
            BES_v2 = (0.35 * norm_margin_risks[idx] +      # 当前边界脆弱性（最重要）
                      0.30 * norm_intrusions[idx] +         # 新类侵入
                      0.20 * norm_drifts[idx] +             # 向新类漂移
                      0.10 * norm_disagreements[idx] +      # 客户端差异
                      0.05 * norm_supports[idx])            # 样本支持度

            # 计算replay count
            replay_count = self._count_replay_samples(class_id)

            # 计算当前accuracy
            current_acc = self._compute_class_accuracy(class_id)

            # 存储数据
            self.task_data[task_id][class_id] = {
                'replay_count': replay_count,
                'low_tail_margin_drop': norm_margin_drops[idx],
                'current_margin_risk': norm_margin_risks[idx],
                'new_intrusion': norm_intrusions[idx],
                'toward_new_drift': norm_drifts[idx],
                'client_disagreement': norm_disagreements[idx],
                'effective_support': norm_supports[idx],
                'BES_v2': BES_v2,
                'acc_before': current_acc,
                'acc_after': None,
                'acc_drop': None,
                # 原始值
                'raw_margin_drop': raw_components[class_id]['low_tail_margin_drop'],
                'raw_margin_risk': raw_components[class_id]['current_margin_risk'],
                'raw_intrusion': raw_components[class_id]['new_intrusion'],
                'raw_drift': raw_components[class_id]['toward_new_drift'],
                'raw_disagreement': raw_components[class_id]['client_disagreement'],
                'raw_support': raw_components[class_id]['effective_support']
            }

            print(f"Class {class_id}: BES_v2={BES_v2:.4f}, Replay={replay_count}, Acc={current_acc:.2f}%")

        # 保存到CSV
        self._save_to_csv(task_id)

    def _compute_raw_components(self, class_id, task_id):
        """计算单个类的原始BES组件（未归一化）"""
        samples = self._collect_class_samples(class_id)

        if len(samples) == 0:
            return {
                'low_tail_margin_drop': 0.0,
                'current_margin_risk': 0.0,
                'new_intrusion': 0.0,
                'toward_new_drift': 0.0,
                'client_disagreement': 0.0,
                'effective_support': 1.0
            }

        # 1. Low-tail Margin Drop (使用20%分位数)
        low_tail_margin_drop = self._compute_low_tail_margin_drop(samples, class_id)

        # 2. Current Margin Risk (当前边界脆弱性)
        current_margin_risk = self._compute_current_margin_risk(samples, class_id)

        # 3. New Intrusion (新类侵入)
        new_intrusion = self._compute_new_intrusion(samples, class_id)

        # 4. Toward-New Drift (向新类方向漂移)
        toward_new_drift = self._compute_toward_new_drift(class_id)

        # 5. Client Disagreement (客户端差异)
        client_disagreement = self._compute_client_disagreement(class_id)

        # 6. Effective Support (样本支持度)
        effective_support = 1.0 / np.sqrt(len(samples) + 1e-8)

        return {
            'low_tail_margin_drop': low_tail_margin_drop,
            'current_margin_risk': current_margin_risk,
            'new_intrusion': new_intrusion,
            'toward_new_drift': toward_new_drift,
            'client_disagreement': client_disagreement,
            'effective_support': effective_support
        }

    def _compute_low_tail_margin_drop(self, samples, class_id):
        """计算低分位数margin的下降"""
        if not hasattr(self.learner, '_old_network') or self.learner._old_network is None:
            return 0.0

        self.learner._network.eval()
        self.learner._old_network.eval()

        old_margins = []
        new_margins = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:
                img = img.unsqueeze(0).cuda()

                # 旧模型的margin
                old_logits = self.learner._old_network(img)['logits'][0]
                old_margin = old_logits[class_id] - torch.max(
                    torch.cat([old_logits[:class_id], old_logits[class_id+1:]])
                )
                old_margins.append(old_margin.item())

                # 新模型的margin
                new_logits = self.learner._network(img)['logits'][0]
                new_margin = new_logits[class_id] - torch.max(
                    torch.cat([new_logits[:class_id], new_logits[class_id+1:]])
                )
                new_margins.append(new_margin.item())

        if not old_margins:
            return 0.0

        # 使用20%分位数
        old_q20 = np.percentile(old_margins, 20)
        new_q20 = np.percentile(new_margins, 20)

        return max(0.0, old_q20 - new_q20)

    def _compute_current_margin_risk(self, samples, class_id):
        """计算当前边界脆弱性（当前old-new margin有多低）"""
        num_new_classes = self.learner._total_classes - self.learner._known_classes
        if num_new_classes == 0:
            return 0.0

        self.learner._network.eval()
        old_new_margins = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:
                img = img.unsqueeze(0).cuda()
                logits = self.learner._network(img)['logits'][0]

                old_class_logit = logits[class_id]
                new_class_logits = logits[self.learner._known_classes:]
                max_new_logit = torch.max(new_class_logits)

                # old-new margin: 旧类logit - 最大新类logit
                old_new_margin = old_class_logit - max_new_logit
                old_new_margins.append(old_new_margin.item())

        if not old_new_margins:
            return 0.0

        # margin越低，风险越高
        # 使用20%分位数
        margin_q20 = np.percentile(old_new_margins, 20)
        m0 = 2.0  # 安全阈值
        risk = max(0.0, m0 - margin_q20)

        return risk

    def _compute_new_intrusion(self, samples, class_id):
        """计算新类侵入程度（保持原逻辑）"""
        num_new_classes = self.learner._total_classes - self.learner._known_classes
        if num_new_classes == 0:
            return 0.0

        self.learner._network.eval()
        intrusions = []

        with torch.no_grad():
            for img, label in samples[:min(100, len(samples))]:
                img = img.unsqueeze(0).cuda()
                logits = self.learner._network(img)['logits'][0]

                new_class_logits = logits[self.learner._known_classes:]
                max_new_logit = torch.max(new_class_logits)
                old_class_logit = logits[class_id]

                intrusion = max(0.0, (max_new_logit - old_class_logit).item())
                intrusions.append(intrusion)

        return np.mean(intrusions) if intrusions else 0.0

    def _compute_toward_new_drift(self, class_id):
        """计算向新类方向的漂移"""
        if not hasattr(self.learner, '_old_network') or self.learner._old_network is None:
            return 0.0

        num_new_classes = self.learner._total_classes - self.learner._known_classes
        if num_new_classes == 0:
            return 0.0

        # 获取旧类和新类的原型
        old_class_proto = self._get_class_prototype(class_id, use_old_network=True)
        cur_class_proto = self._get_class_prototype(class_id, use_old_network=False)

        if old_class_proto is None or cur_class_proto is None:
            return 0.0

        # 获取最近新类的原型
        nearest_new_proto = self._get_nearest_new_class_prototype(cur_class_proto)

        if nearest_new_proto is None:
            return 0.0

        # 计算距离
        dist_old = np.linalg.norm(old_class_proto - nearest_new_proto)
        dist_cur = np.linalg.norm(cur_class_proto - nearest_new_proto)

        # 距离变小 = 向新类靠近 = 风险
        toward_new_drift = max(0.0, dist_old - dist_cur)

        return toward_new_drift

    def _get_class_prototype(self, class_id, use_old_network=False):
        """获取类原型"""
        samples = self._collect_class_samples(class_id)
        if len(samples) == 0:
            return None

        network = self.learner._old_network if use_old_network else self.learner._network
        if network is None:
            return None

        network.eval()
        features = []

        with torch.no_grad():
            for img, label in samples[:min(50, len(samples))]:
                img = img.unsqueeze(0).cuda()
                feat = network(img)['features'][0]
                features.append(feat.cpu().numpy())

        if not features:
            return None

        return np.mean(features, axis=0)

    def _get_nearest_new_class_prototype(self, old_class_proto):
        """获取最近新类的原型"""
        # 从当前任务数据中采样新类原型
        num_old = self.learner._known_classes
        num_total = self.learner._total_classes
        new_class_ids = range(num_old, num_total)

        min_dist = float('inf')
        nearest_proto = None

        for new_class_id in new_class_ids:
            # 简化：从test set采样
            try:
                test_dataset = self.learner.data_manager.get_dataset(
                    np.array([new_class_id]), source="test", mode="test"
                )
                if len(test_dataset) == 0:
                    continue

                loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                features = []

                self.learner._network.eval()
                with torch.no_grad():
                    for _, images, labels in loader:
                        images = images.cuda()
                        feat = self.learner._network(images)['features']
                        features.append(feat.cpu().numpy())
                        if len(features) >= 2:  # 只采样少量
                            break

                if features:
                    new_proto = np.mean(np.concatenate(features, axis=0), axis=0)
                    dist = np.linalg.norm(old_class_proto - new_proto)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_proto = new_proto
            except:
                continue

        return nearest_proto

    def _compute_client_disagreement(self, class_id):
        """计算客户端间原型差异（保持原逻辑）"""
        client_protos = []
        self.learner._network.eval()

        for client_idx in range(self.learner.args['num_users']):
            if not self.learner.retained_ds_all[client_idx]:
                continue

            client_mem = ConcatDataset(self.learner.retained_ds_all[client_idx])
            class_features = []

            with torch.no_grad():
                for i in range(min(len(client_mem), 30)):
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

        return float(np.var(client_protos, axis=0).mean())

    def _collect_class_samples(self, class_id):
        """收集类样本（保持原逻辑）"""
        samples = []
        for client_idx in range(self.learner.args['num_users']):
            if not self.learner.retained_ds_all[client_idx]:
                continue
            client_mem = ConcatDataset(self.learner.retained_ds_all[client_idx])
            for i in range(len(client_mem)):
                try:
                    _, img, label = client_mem[i]
                    if label == class_id:
                        samples.append((img, label))
                except:
                    continue
        return samples

    def _count_replay_samples(self, class_id):
        """统计replay样本数"""
        return len(self._collect_class_samples(class_id))

    def _compute_class_accuracy(self, class_id):
        """计算类accuracy"""
        self.learner._network.eval()
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

    def _save_to_csv(self, task_id):
        """保存到CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_id, data in self.task_data[task_id].items():
                writer.writerow([
                    task_id, class_id, data['replay_count'],
                    data['low_tail_margin_drop'], data['current_margin_risk'],
                    data['new_intrusion'], data['toward_new_drift'],
                    data['client_disagreement'], data['effective_support'],
                    data['BES_v2'],
                    data['acc_before'],
                    data['acc_after'] if data['acc_after'] is not None else '',
                    data['acc_drop'] if data['acc_drop'] is not None else '',
                    data['raw_margin_drop'], data['raw_margin_risk'],
                    data['raw_intrusion'], data['raw_drift'],
                    data['raw_disagreement'], data['raw_support']
                ])

        print(f"Saved diagnostics v2 to {self.csv_path}")
