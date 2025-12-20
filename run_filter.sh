#!/bin/bash

# ========================================================
# run_Filter_batch.sh
# 批量运行 DECOR-Filter 对多个数据集进行排序
# ========================================================

# ------------------ 全局参数 ------------------
MODEL_PATH=""
TOP_K=-1  # -1 表示保留全部结果

# ------------------ 任务列表 ------------------
# 每行格式：输入文件 输出文件 metrics文件
declare -a TASKS=(
)

# --------------------------------------------------
# 可选：激活 Conda 环境（根据你的环境取消注释）
# conda activate your_env_name

echo "🚀 Starting batch reranking for ${#TASKS[@]} tasks..."

# ------------------ 循环执行 ------------------
for task in "${TASKS[@]}"; do
    # 解析输入、输出、metrics 路径
    IFS=' ' read -r INPUT_DATA OUTPUT_FILE METRICS_FILE <<< "$task"

    # 检查输入文件是否存在
    if [ ! -f "$INPUT_DATA" ]; then
        echo "❌ Skip: Input file not found: $INPUT_DATA"
        continue
    fi

    # 创建输出目录
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    mkdir -p "$OUTPUT_DIR"

    # 输出当前任务信息
    echo "📌 Filtering:"
    echo "   Input:  $INPUT_DATA"
    echo "   Output: $OUTPUT_FILE"
    echo "   Metrics: $METRICS_FILE"
    echo "   Model:  $MODEL_PATH"
    echo "   top_k:  $TOP_K"
    echo "--------------------------------------------------"

    # 执行 Python 命令
    python run_filter.py \
      --input_data "$INPUT_DATA" \
      --model_path "$MODEL_PATH" \
      --output_file "$OUTPUT_FILE" \
      --metrics_file "$METRICS_FILE" \
      --top_k "$TOP_K"

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ Success: Reranked data saved to $OUTPUT_FILE"
    else
        echo "❌ Failed: Reranker command failed for $INPUT_DATA"
    fi

    echo "──────────────────────────────────────────"
done

echo "🎉 All reranking tasks completed."