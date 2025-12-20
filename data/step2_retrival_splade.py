
from utils.load_data import load_json,read_json_lines
import json
from sentence_transformers import SparseEncoder
import torch
from tqdm import tqdm


def get_top_k_documents(similarity_scores, docs, k=5):
    # 确保输入是有效的
    if len(docs) == 0:
        return []

    # 将相似度分数转换为一维张量
    scores = similarity_scores.squeeze(0)

    # 获取排序后的索引（降序排列）
    top_indices = torch.argsort(scores, descending=True)

    # 选择前k个文档
    top_k_indices = top_indices[:min(k, len(docs))]

    # 返回结果（文档内容和分数）
    results = []
    for idx in top_k_indices:
        results.append({
            "document": docs[idx],
            "score": scores[idx].item(),
            "rank": len(results) + 1
        })

    return results

if __name__ == '__main__':

    model = SparseEncoder("")

    input_data_path = "."

    output_data_path = ""

    input_data = load_json(input_data_path)

    retrival_data = []
    for i in tqdm(
            range(0, len(input_data)),
            total=len(input_data),
            desc=f"step2 retrival"
    ):
        # 查询和文档
        queries = input_data[i]["question"]
        documents = []

        for doc in input_data[i]["context"]:
            str = ""
            for sen in doc[1]:
                str = str + sen + " "
            documents.append("title: " + doc[0] + " doc:" + str)

        # 生成嵌入向量
        query_embeddings = model.encode_query(queries)
        document_embeddings = model.encode_document(documents)

        # 计算相似度
        similarities = model.similarity(query_embeddings, document_embeddings)

        # 获取前5个最相关的文档
        top_documents = get_top_k_documents(similarities, documents, k=5)

        ctxs = []

        for doc in top_documents:
            title = doc["document"].split(" doc:")[0].replace("title: ","")
            text = doc["document"].split(" doc:")[1]
            score = doc["score"]
            rank = doc["rank"]
            ctxs.append({
                "title": title,
                "text": text,
                "score": score,
                "rank": rank
            })

        retrival_data.append(
            {
                "id": input_data[i]["_id"],
                "question": queries,
                "ctxs": ctxs,
                "gold_answer": input_data[i]["answer"],
                "type": input_data[i]["type"],
                "level": input_data[i]["level"]
            }
        )

    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(retrival_data, f, indent=2, ensure_ascii=False)

