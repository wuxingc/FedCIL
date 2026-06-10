"""
分析BES各组件与forgetting的相关性
帮助诊断哪些组件有效，哪些需要调整
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import argparse


def analyze_component_correlations(csv_path, task_id=None, target='acc_drop'):
    """
    分析各组件与目标变量的相关性

    Args:
        csv_path: CSV文件路径
        task_id: 指定任务ID，None表示分析所有任务
        target: 目标变量，可选：
            - 'acc_drop': 下一任务遗忘 (默认)
            - 'current_task_forgetting': 当前任务遗忘
            - 'old_new_confusion_rate': 旧类被预测为新类的比例
    """
    df = pd.read_csv(csv_path)

    # 筛选有目标数据的记录
    df = df[df[target].notna()].copy()

    if len(df) == 0:
        print(f"No {target} data available in the CSV")
        return

    # 如果指定任务，只分析该任务
    if task_id is not None:
        df = df[df['task_id'] == task_id]
        if len(df) == 0:
            print(f"No {target} data for task {task_id}")
            return

    print("="*80)
    print(f"Analyzing Component Correlations")
    if task_id is not None:
        print(f"Task: {task_id}")
    else:
        print(f"All tasks: {sorted(df['task_id'].unique())}")
    print(f"Target variable: {target}")
    print(f"Number of samples: {len(df)}")
    print("="*80)

    # 定义要分析的组件
    components = {
        'Replay Count': 'replay_count',
    }

    # 自动检测可用的指标列
    if 'BMR' in df.columns:
        components['BMR (Boundary Margin Risk)'] = 'BMR'

    if 'BES_v2' in df.columns:
        components['Low-tail Margin Drop'] = 'low_tail_margin_drop'
        components['Current Margin Risk'] = 'current_margin_risk'
        components['New Intrusion'] = 'new_intrusion'
        components['Toward-New Drift'] = 'toward_new_drift'
        components['Client Disagreement'] = 'client_disagreement'
        components['Effective Support'] = 'effective_support'
        components['BES v2 (Combined)'] = 'BES_v2'

    if 'BES' in df.columns:
        components['margin_drop'] = 'margin_drop'
        components['new_intrusion'] = 'new_intrusion'
        components['prototype_drift'] = 'prototype_drift'
        components['client_disagreement'] = 'client_disagreement'
        components['effective_support'] = 'effective_support'
        components['BES (Combined)'] = 'BES'

    # 计算相关系数
    results = []
    for name, col in components.items():
        if col not in df.columns:
            continue

        values = df[col].values
        target_values = df[target].values

        # Spearman (排序相关)
        spearman_corr, spearman_p = spearmanr(values, target_values)

        # Pearson (线性相关)
        pearson_corr, pearson_p = pearsonr(values, target_values)

        results.append({
            'Component': name,
            'Spearman_ρ': spearman_corr,
            'Spearman_p': spearman_p,
            'Pearson_r': pearson_corr,
            'Pearson_p': pearson_p,
            'Significant': '***' if spearman_p < 0.001 else ('**' if spearman_p < 0.01 else ('*' if spearman_p < 0.05 else ''))
        })

    # 转为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Spearman_ρ', ascending=False, key=abs)

    # 打印结果
    print("\n" + "="*80)
    print(f"CORRELATION WITH {target.upper()}")
    print("="*80)
    print(f"{'Component':<30} {'Spearman ρ':<12} {'p-value':<12} {'Sig.':<5}")
    print("-"*80)

    for _, row in results_df.iterrows():
        print(f"{row['Component']:<30} {row['Spearman_ρ']:>11.3f} {row['Spearman_p']:>11.4f} {row['Significant']:>4}")

    print("="*80)
    print("\n✅ POSITIVE correlation = Higher value → Higher target (Good for risk metric!)")
    print("❌ NEGATIVE correlation = Higher value → Lower target (Bad for risk metric!)")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # 找出表现最好和最差的组件
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS")
    print("="*80)

    best = results_df.iloc[0]
    print(f"\n🏆 Best predictor: {best['Component']}")
    print(f"   Spearman ρ = {best['Spearman_ρ']:.3f} (p={best['Spearman_p']:.4f})")

    worst = results_df.iloc[-1]
    print(f"\n⚠️  Worst predictor: {worst['Component']}")
    print(f"   Spearman ρ = {worst['Spearman_ρ']:.3f} (p={worst['Spearman_p']:.4f})")

    # 检查负相关的组件
    negative_corrs = results_df[results_df['Spearman_ρ'] < 0]
    if len(negative_corrs) > 0:
        print(f"\n❌ Components with NEGATIVE correlation (need fixing):")
        for _, row in negative_corrs.iterrows():
            print(f"   - {row['Component']}: ρ = {row['Spearman_ρ']:.3f}")

    # 比较主指标 vs Replay Count
    main_metric_name = None
    if 'BMR (Boundary Margin Risk)' in results_df['Component'].values:
        main_metric_name = 'BMR (Boundary Margin Risk)'
    elif 'BES v2 (Combined)' in results_df['Component'].values:
        main_metric_name = 'BES v2 (Combined)'
    elif 'BES (Combined)' in results_df['Component'].values:
        main_metric_name = 'BES (Combined)'

    if main_metric_name:
        metric_row = results_df[results_df['Component'] == main_metric_name]
        replay_row = results_df[results_df['Component'] == 'Replay Count']

        if len(metric_row) > 0 and len(replay_row) > 0:
            metric_corr = metric_row.iloc[0]['Spearman_ρ']
            replay_corr = replay_row.iloc[0]['Spearman_ρ']
            improvement = metric_corr - replay_corr

            print("\n" + "="*80)
            print("MAIN METRIC vs BASELINE")
            print("="*80)
            print(f"Replay Count:     ρ = {replay_corr:>7.3f}")
            print(f"{main_metric_name:17s}: ρ = {metric_corr:>7.3f}")
            print(f"Improvement:      Δρ = {improvement:>7.3f}")

            if metric_corr > replay_corr and metric_corr > 0.3:
                print("\n✅ SUCCESS! Main metric significantly better than replay count!")
            elif metric_corr > replay_corr:
                print(f"\n⚠️  Main metric is better but correlation is weak ({metric_corr:.3f})")
                print("   Consider tuning component weights or thresholds")
            else:
                print("\n❌ FAIL! Main metric is worse than replay count")
                print("   Need to fix negative/weak components")

    # 可视化
    visualize_correlations(results_df, task_id, csv_path)

    return results_df


def visualize_correlations(results_df, task_id, csv_path):
    """可视化相关性对比"""
    import os

    fig, ax = plt.subplots(figsize=(10, 6))

    names = results_df['Component'].values
    corrs = results_df['Spearman_ρ'].values
    colors = ['red' if c > 0 else 'blue' for c in corrs]

    # 突出显示BES
    colors = ['darkred' if 'BES' in name else ('steelblue' if 'Replay' in name else c)
              for name, c in zip(names, colors)]

    bars = ax.barh(names, corrs, color=colors, alpha=0.7, edgecolor='black')

    # 标注数值
    for bar, val in zip(bars, corrs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.3f}',
                ha='left' if val > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Spearman Correlation with Forgetting', fontsize=12)
    ax.set_title(f'Component Correlation Analysis' +
                 (f' (Task {task_id})' if task_id else ' (All Tasks)'),
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # 保存
    output_dir = os.path.dirname(csv_path)
    suffix = f'_task_{task_id}' if task_id else '_all_tasks'
    output_path = os.path.join(output_dir, f'component_correlations{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Correlation plot saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze BES component correlations')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to class_diagnostics CSV')
    parser.add_argument('--task', type=int, default=None,
                        help='Task ID to analyze (default: all tasks)')
    parser.add_argument('--target', type=str, default='old_new_confusion_rate',
                        choices=['acc_drop', 'current_task_forgetting', 'old_new_confusion_rate'],
                        help='Target variable to correlate with (default: old_new_confusion_rate)')

    args = parser.parse_args()

    analyze_component_correlations(args.csv, args.task, args.target)
