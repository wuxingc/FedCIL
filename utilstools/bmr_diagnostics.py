"""
简化版 BES 诊断工具 - 单一指标版本
Class-wise Boundary Margin Risk (BMR)

定义：
    BMR_c = -Q_τ(Δ_{c,new}(x))
    其中 Δ_{c,new}(x) = z_c(x) - max_{j∈Y_new} z_j(x)

只用一个指标：旧类样本相对新类的低分位边界margin
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import os
import csv
from collections import defaultdict


class BoundaryMarginRiskDiagnostics:
    """简化版边界风险诊断"""

    def __init__(self, learner, output_dir="run/bmr_diagnostics", tau=0.2):
        self.learner = learner
        self.output_dir = output_dir
        self.tau = tau  # 低分位数（默认20%）
        os.makedirs(output_dir, exist_ok=True)

        self.task_data = defaultdict(dict)
        self.csv_path = os.path.join(output_dir, "class_diagnostics_bmr.csv")
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'task_id', 'class_id', 'replay_count',
                'BMR', 'BMR_normalized',
                'acc_before', 'acc_after', 'acc_drop',
                'acc_before_this_task', 'current_task_forgetting',  # 新增：当前任务遗忘
                'old_new_confusion_rate',  # 新增：旧类被预测为新类的比例
                'raw_low_tail_margin'
            ])

    def compute_bmr_for_task(self, task_id):
        """计算当前任务所有旧类的边界风险"""
        if task_id == 0:
            return

        num_old_classes = self.learner._known_classes
        print(f"\n{'='*80}")
        print(f"Computing Boundary Margin Risk (BMR) for Task {task_id}")
        print(f"Old Classes: 0-{num_old_classes-1}, τ={self.tau}")
        print(f"{'='*80}")

        # 计算所有旧类的原始BMR
        raw_bmr_values = {}
        for class_id in range(num_old_classes):
            raw_bmr = self._compute_class_bmr(class_id, task_id)
            raw_bmr_values[class_id] = raw_bmr

        # 任务内归一化
        bmr_array = np.array(list(raw_bmr_values.values()))
        bmr_min, bmr_max = bmr_array.min(), bmr_array.max()

        if bmr_max - bmr_min < 1e-8:
            # 所有类风险相同，归一化为0
            normalized_bmr = {c: 0.0 for c in raw_bmr_values}
        else:
            normalized_bmr = {
                c: (raw_bmr_values[c] - bmr_min) / (bmr_max - bmr_min + 1e-8)
                for c in raw_bmr_values
            }

        # 保存数据
        for class_id in range(num_old_classes):
            replay_count = self._count_replay_samples(class_id)
            current_acc = self._compute_class_accuracy(class_id)

            # 计算当前任务遗忘（如果有上一任务的accuracy）
            current_task_forgetting = None
            acc_before_this_task = None
            if task_id > 0 and (task_id - 1) in self.task_data:
                if class_id in self.task_data[task_id - 1]:
                    acc_before_this_task = self.task_data[task_id - 1][class_id]['acc_before']
                    current_task_forgetting = max(0.0, acc_before_this_task - current_acc)

            # 计算 old-new confusion rate
            old_new_confusion = self._compute_old_new_confusion(class_id)

            self.task_data[task_id][class_id] = {
                'replay_count': replay_count,
                'BMR': normalized_bmr[class_id],
                'acc_before': current_acc,
                'acc_after': None,
                'acc_drop': None,
                'acc_before_this_task': acc_before_this_task,
                'current_task_forgetting': current_task_forgetting,
                'old_new_confusion_rate': old_new_confusion,
                'raw_bmr': raw_bmr_values[class_id]
            }

            print(f"Class {class_id:2d}: BMR={normalized_bmr[class_id]:.4f} "
                  f"(raw={raw_bmr_values[class_id]:.4f}), "
                  f"Replay={replay_count:3d}, Acc={current_acc:.2f}%, "
                  f"OldNewConf={old_new_confusion:.2f}%"
                  + (f", CurTaskForgetting={current_task_forgetting:.2f}%"
                     if current_task_forgetting is not None else ""))

        self._save_to_csv(task_id)
        print(f"{'='*80}\n")

    def _compute_class_bmr(self, class_id, task_id):
        """
        计算单个类的边界风险

        BMR_c = -Q_τ(Δ_{c,new}(x))
        其中 Δ_{c,new}(x) = z_c(x) - max_{j∈Y_new} z_j(x)
        """
        num_new_classes = self.learner._total_classes - self.learner._known_classes

        if num_new_classes == 0:
            # 没有新类，返回0
            return 0.0

        # 收集replay样本
        samples = self._collect_class_samples(class_id)

        if len(samples) == 0:
            return 0.0

        self.learner._network.eval()
        old_new_margins = []

        with torch.no_grad():
            for img, label in samples:
                img = img.unsqueeze(0).cuda()
                logits = self.learner._network(img)['logits'][0]

                # 旧类logit
                old_logit = logits[class_id]

                # 最大新类logit
                new_logits = logits[self.learner._known_classes:]
                max_new_logit = torch.max(new_logits)

                # old-new margin
                margin = old_logit - max_new_logit
                old_new_margins.append(margin.item())

        if not old_new_margins:
            return 0.0

        # 计算低分位数
        low_tail_margin = np.percentile(old_new_margins, self.tau * 100)

        # BMR = -Q_τ(margin)
        # margin越低（甚至负数），BMR越高，风险越大
        bmr = -low_tail_margin

        return bmr

    def _compute_old_new_confusion(self, class_id):
        """
        计算旧类被预测为新类的比例
        Old-New Confusion Rate = P(pred ∈ Y_new | true = c)
        """
        num_new_classes = self.learner._total_classes - self.learner._known_classes

        if num_new_classes == 0:
            return 0.0

        self.learner._network.eval()

        # 获取该类的测试数据
        test_dataset = self.learner.data_manager.get_dataset(
            np.array([class_id]), source="test", mode="test"
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        total = 0
        confused_to_new = 0

        with torch.no_grad():
            for _, images, labels in test_loader:
                images = images.cuda()
                outputs = self.learner._network(images)['logits']
                preds = torch.argmax(outputs, dim=1)

                # 统计被预测为新类的样本数
                is_new_class = preds >= self.learner._known_classes
                confused_to_new += is_new_class.sum().item()
                total += len(labels)

        if total == 0:
            return 0.0

        return 100.0 * confused_to_new / total

    def _collect_class_samples(self, class_id):
        """收集类样本"""
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
                    data['BMR'], data['BMR'],  # BMR和normalized相同
                    data['acc_before'],
                    data['acc_after'] if data['acc_after'] is not None else '',
                    data['acc_drop'] if data['acc_drop'] is not None else '',
                    data['acc_before_this_task'] if data['acc_before_this_task'] is not None else '',
                    data['current_task_forgetting'] if data['current_task_forgetting'] is not None else '',
                    data['old_new_confusion_rate'],
                    data['raw_bmr']
                ])

        print(f"✅ Saved to {self.csv_path}")
