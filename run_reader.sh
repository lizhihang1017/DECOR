#!/bin/bash

# ========================================================
# run_reader_batch.sh
# 批量运行答案生成任务（如 HotpotQA, TriviaQA 等）
# ========================================================

# ------------------ 全局参数 ------------------
MODEL_PATH=""
DEVICE="cuda:1"

# ------------------ 任务列表 ------------------
# 每行格式：输入文件 输出文件 metrics文件
declare -a TASKS=(

)

# --------------------------------------------------
# 可选：激活 Conda 环境
# conda activate your_env_name

echo "🚀 Starting batch answer generation for ${#TASKS[@]} tasks..."

# ------------------ 循环执行 ------------------
for task in "${TASKS[@]}"; do
    # 解析输入、输出、metrics 路径
    IFS=' ' read -r INPUT OUTPUT METRICS_FILE <<< "$task"

    # 检查输入文件是否存在
    if [ ! -f "$INPUT" ]; then
        echo "❌ Skip: Input file not found: $INPUT"
        continue
    fi

    # 创建输出目录
    OUTPUT_DIR=$(dirname "$OUTPUT")
    mkdir -p "$OUTPUT_DIR"

    # 输出当前任务信息
    echo "📌 Answer Generation:"
    echo "   Input:   $INPUT"
    echo "   Output:  $OUTPUT"
    echo "   Metrics: $METRICS_FILE"
    echo "   Model:   $MODEL_PATH"
    echo "   Device:  $DEVICE"
    echo "   Max new tokens: $MAX_NEW_TOKENS"
    echo "   Temp:    $TEMPERATURE"
    echo "   Top-p:   $TOP_P"
    echo "--------------------------------------------------"

    # 执行 Python 命令
    python run_reader.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --model_path "$MODEL_PATH" \
      --device "$DEVICE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --metrics_file "$METRICS_FILE"

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ Success: Answer generation completed. Results saved to $OUTPUT"
    else
        echo "❌ Failed: Answer generation failed for $INPUT"
    fi

    echo "──────────────────────────────────────────"
done

echo "🎉 All answer generation tasks completed."
