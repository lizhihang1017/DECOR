from sentence_transformers import SentenceTransformer

import sys
import os

from utils.load_data import load_json, read_json_lines
from tqdm import tqdm
import numpy as np
from collections import deque

def get_balanced_negatives(order, positives, segments, min_distance=1):

    # 1. 创建排除集合（正样本及其相邻）
    n = len(segments)
    exclude_set = set()

    # 添加所有正样本
    exclude_set.update(positives)

    # 添加正样本的相邻
    for pos in positives:
        # 左侧相邻
        for i in range(1, min_distance + 1):
            if pos - i >= 0:
                exclude_set.add(pos - i)
        # 右侧相邻
        for i in range(1, min_distance + 1):
            if pos + i < n:
                exclude_set.add(pos + i)

    # 2. 确定候选池（相似度最低的50%）
    mid_point = max(1, len(order) // 2)  # 确保至少1个候选
    candidate_pool = []

    # 遍历排序列表（从相似度最低开始）
    for idx in order:
        # 检查是否达到中点
        if len(candidate_pool) >= mid_point:
            break

        # 检查是否在排除集中
        if idx not in exclude_set:
            candidate_pool.append(idx)

    # 3. 最终过滤：确保最小距离
    final_candidates = []
    for cand in candidate_pool:
        # 计算与所有正样本的最小距离
        min_dist = min(abs(cand - pos) for pos in positives)

        # 满足最小距离要求
        if min_dist > min_distance:
            final_candidates.append(cand)

    return final_candidates  # 返回的是命题列表中的索引


if __name__ == '__main__':

    text_embedder = SentenceTransformer('',device="cuda:1")

    data = load_json("")[:1]

    new_data = []
    for i in tqdm(
            range(0, len(data)),
            total=len(data),
            desc="step6 getting negative samples",
    ):
        sample = data[i]
        embeddings = text_embedder.encode(sample['proposition'])  # 原子命题嵌入(向量化)

        q_emb = text_embedder.encode(sample['question'])  # 问题嵌入(向量化)

        seg_sims = embeddings @ q_emb.T  # 原子命题嵌入与问题做余弦相似度计算(sims为list,包含每个原子命题分别与问题的相似度)
        order = list(map(int, seg_sims.argsort())) # 每个命题与问题的相似度排序位次

        # 计算所有正例句子与问题的相似度
        pos_sentences = sample['positive_ctxs']  # 正样本
        pos_embeddings = text_embedder.encode(pos_sentences)
        pos_sims = (pos_embeddings @ q_emb.T)  # 每个正例的相似度

        # 检查所有正例句子的最小相似度排名
        if len(seg_sims) > 0:
            # 计算最大正例相似度的排名百分比
            max_pos_sim = max(pos_sims)
            rank_pos = np.sum(seg_sims < max_pos_sim)  # 比最大正例相似度低的segment数量
            rank_ratio = rank_pos / (len(seg_sims) - 1) if len(seg_sims) > 1 else 0.0
        else:
            rank_ratio = 0.0

        if rank_ratio < 0.3:
            print(f'similarity of question and positive sentence is too low ({rank_ratio:.4f}). skip this question')
            continue

        # 为每个正例句子生成负例列表
        negative_examples_list = []  # 每个元素对应一个正例的负例索引列表

        for pos_sim in pos_sims:
            # 找出相似度低于当前正例的segments
            mask = seg_sims < pos_sim
            negative_indices = np.where(mask)[0].tolist()

            # 按相似度从低到高排序
            if negative_indices:
                sorted_order = np.argsort(seg_sims[negative_indices])
                negative_indices_sorted = [negative_indices[i] for i in sorted_order]
                negative_scores_sorted = seg_sims[negative_indices_sorted].tolist()
            else:
                negative_indices_sorted = []
                negative_scores_sorted = []

            negative_examples_list.append(negative_indices_sorted)

        sample.pop('ctxs')
        sample['negative_ctxs'] = negative_examples_list

        # 交错合并所有子列表并去重
        merged_negative_indices = []  # 最终合并后的负例索引列表
        seen = set()  # 用于记录已添加的索引，实现去重

        # 创建每个子列表的迭代器队列
        iterators = deque(iter(lst) for lst in negative_examples_list)

        # 继续循环直到所有迭代器都为空
        while iterators:
            # 从队列前端取出一个迭代器
            it = iterators.popleft()

            try:
                # 尝试获取下一个元素
                elem = next(it)

                # 如果元素未重复，则添加到结果中
                if elem not in seen:
                    merged_negative_indices.append(elem)
                    seen.add(elem)

                # 将这个迭代器放回队列末尾（如果还有元素）
                iterators.append(it)
            except StopIteration:
                # 迭代器已耗尽，不再放回队列
                pass
        sample['negative_ctxs_list'] = merged_negative_indices


        new_data.append(sample)


    with open("", 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)










