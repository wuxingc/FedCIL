#!/bin/bash

# BES诊断实验脚本
# 运行FedCIL训练并自动生成BES诊断图

echo "=========================================="
echo "BES Diagnostics Experiment"
echo "=========================================="

# 设置参数
DATASET="cifar100"
TASKS=10
USERS=5
BETA=0.5
COM_ROUND=100
LOCAL_EP=2
MEM_SIZE=500
SEED=2023

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Tasks: $TASKS"
echo "  Users: $USERS"
echo "  Beta: $BETA"
echo "  Memory Size: $MEM_SIZE"
echo ""

# 运行训练
echo "Step 1: Running training with BES diagnostics..."
python main.py \
  --dataset $DATASET \
  --method bbb \
  --tasks $TASKS \
  --num_users $USERS \
  --com_round $COM_ROUND \
  --local_ep $LOCAL_EP \
  --beta $BETA \
  --gpu 0 \
  --seed $SEED \
  --mem_size $MEM_SIZE \
  --w_old 3.0 \
  --w_new 1.0 \
  --tau_old 0.9 \
  --tau_new 1.1 \
  --local_lr 0.05 \
  --weight_decay 1e-5 \
  --local_bs 128

echo ""
echo "Step 2: Generating diagnostic plots..."

# 检查CSV文件是否存在
CSV_FILE="run/bes_diagnostics/class_diagnostics.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: $CSV_FILE not found!"
    exit 1
fi

# 为每个有forgetting数据的任务生成图
# 通常从task 2开始有forgetting数据（task 1结束后观测到task 0的forgetting）
for TASK in {2..9}; do
    echo "  Plotting Task $TASK..."
    python utilstools/plot_bes_diagnostics.py \
        --csv $CSV_FILE \
        --task $TASK \
        --correlation
done

echo ""
echo "=========================================="
echo "Diagnostics Complete!"
echo "Results saved in: run/bes_diagnostics/"
echo "=========================================="
