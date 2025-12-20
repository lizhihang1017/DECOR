
import torch
import sys
import os
import json
from utils.load_data import load_json, read_json_lines
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


if __name__ == '__main__':

    device = "cuda:1"

    # 初始化嵌入模型
    text_embedder = SentenceTransformer('',device=device)

    # 读取数据
    input_file = 
    output_file = 

    data = list(read_json_lines(input_file))
    processed_data = []

    for sample in tqdm(data, desc="Generating pos-neg pairs"):
        # 提取关键数据
        propositions = sample["proposition"]
        pos_indices = sample["dpr_instance"]["positive_ctxs"]
        neg_indices = sample["dpr_instance"]["neg_samples"]

        try:
            # 验证索引是否在范围内
            max_index = len(propositions) - 1

            # 过滤掉无效的正样本索引
            valid_pos_indices = [i for i in pos_indices if isinstance(i, int) and 0 <= i <= max_index]

            # 过滤掉无效的负样本索引
            valid_neg_indices = [i for i in neg_indices if isinstance(i, int) and 0 <= i <= max_index]

            # 如果有效正样本或负样本为空，跳过处理
            if not valid_pos_indices or not valid_neg_indices:
                sample["neg_pos_pairs"] = []
                continue

            # 获取正负样本文本
            pos_texts = [propositions[i] for i in pos_indices]
            neg_texts = [propositions[i] for i in neg_indices]

            # 如果没有负样本则跳过
            if not neg_texts:
                sample["neg_pos_pairs"] = []
                # processed_data.append(sample)
                continue

            # 生成文本嵌入
            pos_embeddings = text_embedder.encode(pos_texts, convert_to_tensor=True, device=device)
            neg_embeddings = text_embedder.encode(neg_texts, convert_to_tensor=True, device=device)

            # 确保所有张量都在同一个设备上
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)

            # 计算余弦相似度矩阵
            cos_sim = util.pytorch_cos_sim(neg_embeddings, pos_embeddings)

            # 为每个负样本找到最相似的正样本
            pairs = []
            for i, neg_idx in enumerate(neg_indices):
                # 获取当前负样本的相似度向量
                sim_vector = cos_sim[i]

                # 找到最相似的正样本索引
                most_similar_idx = torch.argmax(sim_vector).item()

                # 获取对应的文本
                neg_text = propositions[neg_idx]
                pos_text = propositions[pos_indices[most_similar_idx]]

                pairs.append((pos_indices[most_similar_idx], neg_idx))

            # 将配对结果添加到样本数据中
            sample["neg_pos_pairs"] = pairs

            processed_data.append(sample)

        except Exception as e:
            print(f"处理样本时出错: {e}")



    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete! Results saved to {output_file}")