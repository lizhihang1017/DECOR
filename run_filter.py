import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import json
import torch
from accelerate import Accelerator


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_contriever_scores(model, tokenizer, data_row, device, top_k):
    if len(data_row['retrieved_docs']) == 0:
        return [], 0
    if top_k == -1:
        corpus = [data['text'] for data in data_row['retrieved_docs']]
    else:
        corpus = [data['text'] for data in data_row['retrieved_docs'][:top_k]]

    query = data_row['question']
    inputs = tokenizer([query] + corpus, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 估算FLOPs - 基于参数数量和序列长度
    # 这是一个简化的估算，实际FLOPs可能有所不同
    num_params = sum(p.numel() for p in model.parameters())
    seq_length = inputs['input_ids'].shape[1]
    batch_size = inputs['input_ids'].shape[0]

    # 简化的FLOPs估算公式: 2 * 参数数量 * 序列长度 * batch_size
    # 这是一个非常粗略的估算，实际值可能有所不同
    estimated_flops = 2 * num_params * seq_length * batch_size

    # Compute token embeddings
    outputs = model(**inputs)

    embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
    scores = []
    for i in range(len(corpus)):
        scores.append((embeddings[0] @ embeddings[i+1]).item())
    return scores, estimated_flops


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--input_data", dest='input_data', required=True)
    argparse.add_argument("--model_type", dest='model_type', required=False,
                          choices=['dpr', 'bm25', 'facebook/contriever-msmarco', 'facebook/contriever'])
    argparse.add_argument("--model_path", dest='model_path', required=False)
    argparse.add_argument("--output_file", dest='output_file', type=str, required=True)
    argparse.add_argument("--device", dest='device', default=1, type=int)
    argparse.add_argument("--top_k", dest='top_k', default=30, type=int)
    argparse.add_argument("--metrics_file", dest='metrics_file', default="metrics.json", type=str)

    args = argparse.parse_args()
    print(args)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    device = "cuda:0"
    input_data_df = pd.read_json(args.input_data)
    print(input_data_df.columns)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path,device_map="cuda:0")
    # model.to(device)
    model.eval()

    total_flops = 0
    total_forward_time = 0
    contriever_scores = []
    start_time = time.time()

    for _, data in tqdm(input_data_df.iterrows(), total=len(input_data_df)):
        forward_start = time.time()
        scores, flops = get_contriever_scores(model, tokenizer, data, device, top_k=args.top_k)
        forward_end = time.time()

        contriever_scores.append(scores)
        total_flops += flops
        total_forward_time += (forward_end - forward_start)

    total_time = time.time() - start_time

    input_data_df['compressor_scores'] = contriever_scores
    input_data_df.to_json(args.output_file, orient='records')

    # Calculate metrics
    gflops = total_flops / (total_forward_time * 1e9)  # GFLOPS

    metrics = {
        "total_runtime_seconds": total_time,
        "total_forward_pass_seconds": total_forward_time,
        "total_flops": total_flops,
        "average_gflops": gflops,
        "number_of_queries": len(input_data_df),
        "average_forward_time_per_query": total_forward_time / len(input_data_df)
    }

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {args.metrics_file}")
    print(metrics)

if __name__ == "__main__":
    main()