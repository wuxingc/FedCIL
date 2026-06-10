"""
BES诊断结果后处理脚本
从CSV中计算forgetting并重新生成完整的CSV
"""
import pandas as pd
import numpy as np


def compute_forgetting_from_csv(input_csv, output_csv=None):
    """
    从原始CSV计算forgetting

    逻辑：
    - Task t 结束时记录的 acc_before 就是该任务结束时旧类的accuracy
    - Task t+1 结束时，相同class的 acc_before 就是 Task t 的 acc_after
    - acc_drop = Task t的acc_before - Task t的acc_after
    """
    df = pd.read_csv(input_csv)

    # 按class_id和task_id排序
    df = df.sort_values(['class_id', 'task_id'])

    # 为每个类计算forgetting
    for class_id in df['class_id'].unique():
        class_df = df[df['class_id'] == class_id].sort_values('task_id')

        # 对于每个任务，下一个任务的acc_before就是它的acc_after
        for i in range(len(class_df) - 1):
            current_idx = class_df.index[i]
            next_idx = class_df.index[i + 1]

            acc_before = df.loc[current_idx, 'acc_before']
            acc_after = df.loc[next_idx, 'acc_before']  # 下一任务的acc_before

            df.loc[current_idx, 'acc_after'] = acc_after
            df.loc[current_idx, 'acc_drop'] = acc_before - acc_after

    # 保存
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_forgetting.csv')

    df.to_csv(output_csv, index=False)

    print(f"✓ Forgetting data computed successfully!")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_csv}")

    # 统计
    with_forgetting = df['acc_drop'].notna().sum()
    total = len(df)
    print(f"\n  Total records: {total}")
    print(f"  With forgetting data: {with_forgetting}")
    print(f"  Tasks with forgetting: {sorted(df[df['acc_drop'].notna()]['task_id'].unique())}")

    return output_csv


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compute_forgetting.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    compute_forgetting_from_csv(input_csv, output_csv)
