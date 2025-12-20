import sys
import os
import argparse
from utils.load_data import load_json, read_json_lines
from utils.prompt import HotpotQA_PROMPT,MuSiQue_PROMPT,NQ_PROMPT,TQA_PROMPT
import transformers
import torch
import json
from tqdm import tqdm
import time

def get_top_k_sent(retrieved_docs, compressor_scores, k):
    """
    返回 compressor_scores 排序后得分最高的前 K 个文档的 text。

    参数:
        retrieved_docs (list of dict): 包含文档的列表，每个文档是 {'text': '...'} 的形式
        compressor_scores (list of float): 每个文档对应的得分，与 retrieved_docs 一一对应
        k (int): 要返回的前 K 个文档数量

    返回:
        list of str: 得分最高的前 K 个文档的 text 内容
    """
    # 将文档和得分配对，并按得分从高到低排序
    scored_docs = sorted(zip(retrieved_docs, compressor_scores), key=lambda x: x[1], reverse=True)

    # 提取前 K 个文档的 text
    top_k_texts = [doc['text'] for doc, score in scored_docs[:k]]
    top_k_texts = "".join(top_k_texts)
    return top_k_texts

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Process Q&A with Llama model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--model_path', type=str, default="",
                        help='Path to model directory')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=10,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--metrics_file', type=str, default="metrics.json",
                        help='Path to save metrics file')

    args = parser.parse_args()

    # 使用命令行参数
    input_data_path = args.input
    output_data_path = args.output
    model_name_or_path = args.model_path
    device = args.device
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p

    # 加载输入数据
    data = load_json(input_data_path)

    # 初始化模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"pad_token_id": tokenizer.eos_token_id},
        device_map=device,
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # FLOPs 估算准备
    num_params = sum(p.numel() for p in model.parameters())

    total_flops = 0
    total_forward_time = 0
    start_time = time.time()

    if "HotpotQA" in input_data_path:
        use_prompt = HotpotQA_PROMPT
    elif "musique" in input_data_path:
        use_prompt = MuSiQue_PROMPT
    elif "NQ" in input_data_path:
        use_prompt = NQ_PROMPT
    elif "TQA" in input_data_path:
        use_prompt = TQA_PROMPT
    else:
        use_prompt = HotpotQA_PROMPT

    # 处理每个样本
    for i in tqdm(range(len(data)), desc="test Q&A with llama-3-8B-Instruct....."):
        sample = data[i]
        question = sample["question"]

        # if 'BM25' in input_data_path:
        #     sample["text"] = get_top_k_sent(sample["retrieved_docs"], sample["bm25_scores"], k=10)

        # sent_list = [it['text'] for it in sample['scored_sentences'][:10]]
        # sample["text"] = "".join(sent_list)

        # if 'RECOMP-Extr' in input_data_path:
        #     sample["text"] = get_top_k_sent(sample["retrieved_docs"], sample["compressor_scores"], k=5)
        # if 'CPC' in input_data_path:
        #     sample["text"] = sample["text"][:10]
        # if 'selector_analys/CPC' in input_data_path:
        #     sample["text"] = sample['selected_context']
        # if 'Provence' in input_data_path:
        #     text = ""
        #     for te in sample["text_list"]:
        #         text += te
        #     sample["text"] = text
        if 'RECOMP-Abs' in input_data_path:
            text = ""
            for te in sample["text_list"]:
                text += te
            sample["text"] = text

        if 'lmchunker' in input_data_path or 'MetaChunking' in input_data_path:
            text = ""
            for chunk in sample["top_chunks"][:3]:
                text = text + chunk['chunk'] + " "
            sample["text"] = text


        try:
            text = sample["text"]
        except Exception as e:
            print(f"Error building context: {e}")
            text = ""


        prompt = use_prompt.format(retrieved_documents=text, question=question)

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": prompt},
        ]

        # if "2wikimultihop" in input_data_path:
        #
        #     # Format prompt
        #     messages = [{
        #         "role": "system",
        #         "content": f"Context information is below.\n---------------------\n{text}\n---------------------\nGiven the context information and not prior knowledge, answer the query. Do not provide any explanation."
        #     },
        #         {
        #             "role": "user",
        #             "content": f"Query: {question}\nAnswer: "
        #         }]


        # ====== forward 开始 ======
        forward_start = time.time()
        outputs = pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        forward_end = time.time()
        # ====== forward 结束 ======

        # FLOPs 估算：2 * 参数量 * 序列长度 * batch_size
        # 这里 batch_size=1，序列长度取输入 prompt 的 token 长度 + 输出长度
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        seq_length = input_ids.shape[1] + max_new_tokens
        flops = 2 * num_params * seq_length * 1

        total_flops += flops
        total_forward_time += (forward_end - forward_start)

        try:
            generated_content = outputs[0]["generated_text"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            generated_content = "Error: Failed to extract answer"

        sample["generated_answer"] = generated_content

    total_time = time.time() - start_time

    # 保存结果
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 计算指标
    gflops = total_flops / (total_forward_time * 1e9) if total_forward_time > 0 else 0
    metrics = {
        "total_runtime_seconds": total_time,
        "total_forward_pass_seconds": total_forward_time,
        "total_flops": total_flops,
        "average_gflops": gflops,
        "number_of_queries": len(data),
        "average_forward_time_per_query": total_forward_time / len(data) if len(data) > 0 else 0
    }

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {args.metrics_file}")
    print(metrics)


if __name__ == "__main__":
    main()
