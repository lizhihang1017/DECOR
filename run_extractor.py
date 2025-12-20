import sys
import os
import argparse
import time
import json
import re
import torch
import transformers
from tqdm import tqdm

from utils.load_data import load_json, read_json_lines
from utils.prompt import dake_prompt
from utils.eval_utils import single_ans_em


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Process NQ dataset with Qwen model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to Qwen model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top documents to consider')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--metrics_file', type=str, default="metrics.json",
                        help='Path to output metrics JSON file')

    args = parser.parse_args()

    # 使用命令行参数
    input_data_path = args.input
    output_data_path = args.output
    model_name_qwen = args.model_path
    device = args.device
    batch_size = args.batch_size
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    metrics_file = args.metrics_file


    # 加载输入数据
    data = load_json(input_data_path)

    # 加载Qwen模型和分词器
    model_qwen = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_qwen,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer_qwen = transformers.AutoTokenizer.from_pretrained(model_name_qwen)

    # 计算模型参数数量（用于FLOPs估算）
    num_params = sum(p.numel() for p in model_qwen.parameters())

    # 初始化指标
    total_time = 0
    total_forward_time = 0
    total_flops = 0
    total_queries = len(data)

    # 开始计时
    start_time = time.time()

    # 将数据分成批次
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    for batch in tqdm(batches, desc="Processing batches"):
        # 准备批处理数据
        batch_questions = []
        batch_scored_docs = []
        batch_orig_indices = []  # 保存原始数据索引
        batch_inputs_list = []  # 存储所有输入文本

        for i, sample in enumerate(batch):
            question = sample["question"]

            retrieved_docs = sample['retrieved_docs']
            compressor_scores = sample['compressor_scores']
            scored_docs = sorted(zip(retrieved_docs, compressor_scores), key=lambda x: x[1], reverse=True)[:top_k]
            sents = ""
            for j, (doc, score) in enumerate(scored_docs):
                sents += f"[{j}] {doc['text']}\n"

            batch_questions.append(question)
            batch_scored_docs.append(scored_docs)
            batch_orig_indices.append(i)

            # 构建提示
            prompt = dake_prompt.format(question=question, num=len(scored_docs), context=sents)
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": prompt},
            ]

            # 处理输入文本
            text = tokenizer_qwen.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            batch_inputs_list.append(text)

        # 批量编码
        batch_inputs = tokenizer_qwen(
            batch_inputs_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # 估算FLOPs - 基于参数数量和序列长度
        # 这是一个简化的估算，实际FLOPs可能有所不同
        seq_length = batch_inputs.input_ids.shape[1]
        batch_size_actual = batch_inputs.input_ids.shape[0]

        # 简化的FLOPs估算公式: 2 * 参数数量 * 序列长度 * batch_size
        estimated_flops = 2 * num_params * seq_length * batch_size_actual

        # 批量生成
        forward_start = time.time()
        with torch.no_grad():
            batch_outputs = model_qwen.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer_qwen.eos_token_id,
                pad_token_id=tokenizer_qwen.eos_token_id,
            )
        forward_end = time.time()

        # 更新指标
        total_forward_time += (forward_end - forward_start)
        total_flops += estimated_flops

        # 解码每个样本的响应
        batch_responses = []
        for j, output in enumerate(batch_outputs):
            # 跳过输入部分，只解码生成的部分
            response = tokenizer_qwen.decode(
                output[len(batch_inputs.input_ids[j]):],
                skip_special_tokens=True
            )
            batch_responses.append(response)

        # 处理每个样本的响应
        for j, response in enumerate(batch_responses):
            orig_idx = batch_orig_indices[j]
            scored_docs = batch_scored_docs[j]
            response = response.split('Final Selection:')[-1]
            try:
                if "No answer" in response:
                    selected_idxs = []
                else:
                    matches = re.findall(r'\[([\d, ]+)\]', response)
                    selected_idxs = set()
                    for match in matches:
                        for num in match.split(','):
                            num = num.strip()
                            if num.isdigit():
                                selected_idxs.add(int(num))
            except Exception as e:
                print(f"Error parsing indexes: {e}")
                selected_idxs = []

            # 构建最终上下文
            try:
                text = ""
                for idx in selected_idxs:
                    if idx < len(scored_docs):
                        text += scored_docs[idx][0]['text'] + " "
            except Exception as e:
                print(f"Error building context: {e}")
                text = ""


            # 存储结果回原始数据
            batch[orig_idx]["response"] = response
            batch[orig_idx]["text"] = text

    # 计算总时间
    total_time = time.time() - start_time

    # 计算GFLOPS
    gflops = total_flops / (total_forward_time * 1e9) if total_forward_time > 0 else 0

    # 准备指标数据
    metrics = {
        "total_runtime_seconds": total_time,
        "total_forward_pass_seconds": total_forward_time,
        "total_flops": total_flops,
        "average_gflops": gflops,
        "number_of_queries": total_queries,
        "average_forward_time_per_query": total_forward_time / total_queries if total_queries > 0 else 0
    }

    # 保存输出数据
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 保存指标数据
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Processing completed. Metrics saved to {metrics_file}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()