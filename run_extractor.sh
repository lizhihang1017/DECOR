#!/bin/bash

# ========================================================
# run_extractor_batch.sh
# 批量运行 DECOR-extractor 对多个 filtered 数据进行语句选择
# ========================================================

# ------------------ 全局参数 ------------------
MODEL_PATH=""
DEVICE="cuda:1"
BATCH_SIZE=1
TOP_K=20          # 保留前 K 个相关语句
MAX_NEW_TOKENS=2048
TEMPERATURE=0.6
TOP_P=0.9

# ------------------ 任务列表 ------------------
# 每行格式：输入文件 输出文件 metrics文件
declare -a TASKS=(

)

# --------------------------------------------------
# 可选：激活 Conda 环境
# conda activate your_env_name

echo "🚀 Starting batch sentence selection for ${#TASKS[@]} tasks..."

# ------------------ 循环执行 ------------------
for task in "${TASKS[@]}"; do
    # 解析三个字段
    IFS=' ' read -r INPUT OUTPUT METRICS_FILE <<< "$task"

    # 检查输入文件
    if [ ! -f "$INPUT" ]; then
        echo "❌ Skip: Input file not found: $INPUT"
        continue
    fi

    # 创建输出目录
    OUTPUT_DIR=$(dirname "$OUTPUT")
    mkdir -p "$OUTPUT_DIR"

    # 输出当前任务信息
    echo "📌 Sentence Extractor:"
    echo "   Input:  $INPUT"
    echo "   Output: $OUTPUT"
    echo "   Metrics: $METRICS_FILE"
    echo "   Model:  $MODEL_PATH"
    echo "   Top-k:  $TOP_K"
    echo "   Device: $DEVICE"
    echo "--------------------------------------------------"

    # 执行 Python 命令
    python run_extractor.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --model_path "$MODEL_PATH" \
      --device "$DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --top_k "$TOP_K" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --metrics_file "$METRICS_FILE"

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ Success: Selected results saved to $OUTPUT"
    else
        echo "❌ Failed: Selector command failed for $INPUT"
    fi

    echo "──────────────────────────────────────────"
done

echo "🎉 All sentence selection tasks completed."