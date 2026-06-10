"""
BES诊断结果可视化脚本
生成两个对比图：
(a) Replay Count vs Forgetting
(b) BES vs Forgetting
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import argparse
import os


def plot_bes_diagnostics(csv_path, task_id, output_dir=None, target='acc_drop'):
    """
    绘制诊断图

    Args:
        csv_path: CSV文件路径
        task_id: 要可视化的任务ID
        output_dir: 输出目录，默认与CSV同目录
        target: 目标变量，可选 acc_drop / current_task_forgetting / old_new_confusion_rate
    """
    # 读取数据
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise KeyError(f"CSV does not contain target column: {target}")

    # 筛选指定任务的数据
    task_df = df[df['task_id'] == task_id].copy()

    # 只保留有目标变量数据的行
    task_df = task_df[task_df[target].notna()].copy()

    if len(task_df) == 0:
        print(f"Warning: No {target} data available for task {task_id}")
        print(f"Available tasks with {target} data: {df[df[target].notna()]['task_id'].unique()}")
        return

    # 按class_id排序
    task_df = task_df.sort_values('class_id')

    # 提取数据
    class_ids = task_df['class_id'].values
    replay_counts = task_df['replay_count'].values

    # 自动检测指标列名（支持v1的'BES'、v2的'BES_v2'、简化版的'BMR'）
    if 'BMR' in task_df.columns:
        bes_scores = task_df['BMR'].values
        bes_label = 'BMR'
        bes_full_name = 'Boundary Margin Risk'
    elif 'BES_v2' in task_df.columns:
        bes_scores = task_df['BES_v2'].values
        bes_label = 'BES v2'
        bes_full_name = 'Boundary Erosion Score v2'
    elif 'BES' in task_df.columns:
        bes_scores = task_df['BES'].values
        bes_label = 'BES'
        bes_full_name = 'Boundary Erosion Score'
    else:
        raise KeyError("CSV does not contain 'BES', 'BES_v2', or 'BMR' column")

    target_labels = {
        'acc_drop': ('Forgetting', 'Accuracy Drop (%)', 'accuracy_drop'),
        'current_task_forgetting': ('Current-Task Forgetting', 'Current-Task Forgetting (%)', 'current_task_forgetting'),
        'old_new_confusion_rate': ('Old-New Confusion', 'Old-New Confusion Rate (%)', 'old_new_confusion')
    }
    target_name, target_ylabel, target_suffix = target_labels.get(target, (target, target, target))

    target_values = task_df[target].values

    # 计算相关系数
    corr_replay, _ = spearmanr(replay_counts, target_values)
    corr_bes, _ = spearmanr(bes_scores, target_values)

    print(f"\n=== Task {task_id} Diagnostics ===")
    print(f"Target: {target_name}")
    print(f"Number of old classes: {len(class_ids)}")
    print(f"Spearman correlation (Replay Count, {target_name}): {corr_replay:.3f}")
    print(f"Spearman correlation ({bes_label}, {target_name}): {corr_bes:.3f}")
    print(f"{bes_label} improvement: {(corr_bes - corr_replay):.3f}")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 子图(a): Replay Count vs Forgetting ==========
    ax1_twin = ax1.twinx()

    # 柱状图：replay count
    bars1 = ax1.bar(class_ids, replay_counts, alpha=0.6, color='steelblue',
                     label='Replay Count', width=0.6)

    # 折线图：目标变量
    line1 = ax1_twin.plot(class_ids, target_values, 'ro-', linewidth=2,
                           markersize=6, label=target_name)

    ax1.set_xlabel('Old Class ID', fontsize=12)
    ax1.set_ylabel('Replay Count', fontsize=12, color='steelblue')
    ax1_twin.set_ylabel(target_ylabel, fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.set_title(f'(a) Replay Count vs. {target_name}\n' +
                  r'$\rho$' + f'(Replay, {target_name}) = {corr_replay:.3f}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # ========== 子图(b): BES vs Forgetting ==========
    ax2_twin = ax2.twinx()

    # 柱状图：BES
    bars2 = ax2.bar(class_ids, bes_scores, alpha=0.6, color='orange',
                     label=bes_label, width=0.6)

    # 折线图：目标变量
    line2 = ax2_twin.plot(class_ids, target_values, 'ro-', linewidth=2,
                           markersize=6, label=target_name)

    ax2.set_xlabel('Old Class ID', fontsize=12)
    ax2.set_ylabel(f'{bes_label}', fontsize=12, color='orange')
    ax2_twin.set_ylabel(target_ylabel, fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.set_title(f'(b) {bes_label} vs. {target_name}\n' +
                  r'$\rho$' + f'({bes_label}, {target_name}) = {corr_bes:.3f}',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    output_path = os.path.join(output_dir, f'bes_diagnostics_task_{task_id}_{target_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    # 打印详细数据表
    print("\n=== Detailed Data ===")
    print(f"{'Class':<8} {'Replay':<10} {bes_label:<10} {target_name:<12}")
    print("-" * 50)
    for i in range(len(class_ids)):
        print(f"{int(class_ids[i]):<8} {int(replay_counts[i]):<10} "
              f"{bes_scores[i]:<10.4f} {target_values[i]:<12.2f}")


def plot_correlation_comparison(csv_path, task_id, output_dir=None):
    """
    绘制不同指标与forgetting的相关性对比图

    Args:
        csv_path: CSV文件路径
        task_id: 要可视化的任务ID
        output_dir: 输出目录
    """
    df = pd.read_csv(csv_path)
    task_df = df[(df['task_id'] == task_id) & (df['acc_drop'].notna())].copy()

    if len(task_df) == 0:
        print(f"No data for task {task_id}")
        return

    acc_drops = task_df['acc_drop'].values

    # 计算各指标的相关系数
    metrics = {
        'Replay Count': task_df['replay_count'].values,
        'Margin Drop': task_df['margin_drop'].values,
        'New Intrusion': task_df['new_intrusion'].values,
        'Prototype Drift': task_df['prototype_drift'].values,
        'Effective Support': task_df['effective_support'].values,
        'BES (Combined)': task_df['BES'].values
    }

    correlations = {}
    for name, values in metrics.items():
        corr, _ = spearmanr(values, acc_drops)
        correlations[name] = corr

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(correlations.keys())
    values = list(correlations.values())
    colors = ['steelblue'] * (len(names) - 1) + ['red']  # BES用红色突出

    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')

    # 在柱子上标注数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Spearman Correlation with Forgetting', fontsize=12)
    ax.set_title(f'Correlation Comparison (Task {task_id})',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    output_path = os.path.join(output_dir, f'correlation_comparison_task_{task_id}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation comparison saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot BES diagnostics')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to class_diagnostics.csv')
    parser.add_argument('--task', type=int, required=True,
                        help='Task ID to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--correlation', action='store_true',
                        help='Also plot correlation comparison')
    parser.add_argument('--target', type=str, default='acc_drop',
                        choices=['acc_drop', 'current_task_forgetting', 'old_new_confusion_rate'],
                        help='Target variable to plot')

    args = parser.parse_args()

    # 绘制主诊断图
    plot_bes_diagnostics(args.csv, args.task, args.output_dir, args.target)

    # 可选：绘制相关性对比图
    if args.correlation:
        plot_correlation_comparison(args.csv, args.task, args.output_dir)
