#!/bin/bash

# BES v2 快速验证脚本
# 运行小规模测试验证v2是否工作

echo "=========================================="
echo "BES v2 Quick Validation Test"
echo "=========================================="
echo ""
echo "Changes made:"
echo "  ✅ methods/B.py now uses BESDiagnosticsV2"
echo "  ✅ Output directory: run/bes_diagnostics_v2/"
echo ""
echo "Running quick test (3 tasks, 3 users, 5 rounds)..."
echo ""

# 小规模快速测试
DATASET="cifar100"
TASKS=3
USERS=3
BETA=0.5
COM_ROUND=5  # 非常少的轮数用于快速验证
LOCAL_EP=2
MEM_SIZE=100
SEED=2023

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
echo "=========================================="
echo "Checking results..."
echo "=========================================="

CSV_V2="run/bes_diagnostics_v2/class_diagnostics_v2.csv"

if [ -f "$CSV_V2" ]; then
    echo "✅ CSV file created: $CSV_V2"
    echo ""
    echo "CSV header:"
    head -1 "$CSV_V2"
    echo ""
    echo "First data row:"
    head -2 "$CSV_V2" | tail -1
    echo ""
    echo "Number of records:"
    wc -l "$CSV_V2"
    echo ""

    # 检查关键列是否有数据
    echo "Checking key columns..."

    # 检查是否有非零的组件值
    awk -F',' 'NR>1 {
        if ($4 != 0) print "✅ low_tail_margin_drop has non-zero values"
        if ($5 != 0) print "✅ current_margin_risk has non-zero values"
        if ($6 != 0) print "✅ new_intrusion has non-zero values"
        if ($7 != 0) print "✅ toward_new_drift has non-zero values"
        if ($10 != 0 && $10 != "") print "✅ BES_v2 has non-zero values"
        exit
    }' "$CSV_V2"

    echo ""
    echo "=========================================="
    echo "Quick Test Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Compute forgetting:"
    echo "   python utilstools/compute_forgetting.py \\"
    echo "       run/bes_diagnostics_v2/class_diagnostics_v2.csv \\"
    echo "       run/bes_diagnostics_v2/class_diagnostics_v2_with_forgetting.csv"
    echo ""
    echo "2. Analyze components:"
    echo "   python utilstools/analyze_bes_components.py \\"
    echo "       --csv run/bes_diagnostics_v2/class_diagnostics_v2_with_forgetting.csv \\"
    echo "       --task 1"
    echo "=========================================="
else
    echo "❌ Error: CSV file not created at $CSV_V2"
    echo "Please check error messages above"
fi
