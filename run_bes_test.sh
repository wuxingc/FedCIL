#!/bin/bash

# BES诊断快速测试脚本（小规模）
# 用于验证代码是否正常工作

echo "=========================================="
echo "BES Diagnostics Quick Test"
echo "=========================================="

# 小规模参数（快速验证）
DATASET="cifar100"
TASKS=5
USERS=3
BETA=0.5
COM_ROUND=10  # 只跑10轮
LOCAL_EP=2
MEM_SIZE=100  # 减小memory
SEED=2023

echo "Quick Test Configuration:"
echo "  Dataset: $DATASET"
echo "  Tasks: $TASKS (reduced)"
echo "  Users: $USERS (reduced)"
echo "  Com Rounds: $COM_ROUND (reduced)"
echo "  Memory Size: $MEM_SIZE (reduced)"
echo ""
echo "This is a quick test to verify the code works."
echo "For full experiments, use run_bes_diagnostics.sh"
echo ""

# 运行训练
echo "Running quick test..."
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
echo "Checking results..."

CSV_FILE="run/bes_diagnostics/class_diagnostics.csv"
if [ -f "$CSV_FILE" ]; then
    echo "✓ CSV file created successfully"
    echo "  Location: $CSV_FILE"
    echo ""
    echo "First few lines:"
    head -n 10 $CSV_FILE
    echo ""

    # 尝试生成一个图
    echo "Generating test plot for Task 2..."
    python utilstools/plot_bes_diagnostics.py \
        --csv $CSV_FILE \
        --task 2

    echo ""
    echo "=========================================="
    echo "Quick Test Complete!"
    echo "If you see the plot, the code is working."
    echo "Now you can run the full experiment with:"
    echo "  bash run_bes_diagnostics.sh"
    echo "=========================================="
else
    echo "✗ Error: CSV file not created"
    echo "Please check the error messages above"
fi
