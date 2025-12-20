import sys

from utils.load_data import load_json, read_json_lines


data = {
    "id": "5a7a06935542990198eaf050",
    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
    "positive_ctxs": [
        "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
        "First for Women is a woman's magazine published by Bauer Media Group in the USA."
    ],
    "gold_answers": "Arthur's Magazine",
    "negative_ctxs": [
        [33, 19, 15, 13, 27, 34, 14, 23, 26, 12, 32, 22, 17, 28, 30, 21, 29, 24, 31, 10, 20, 25, 35, 11, 18, 16, 9, 8,
         7, 4, 3, 6, 1, 2, 0],
        [33, 19, 15, 13, 27, 34, 14, 23, 26, 12, 32, 22, 17, 28, 30, 21, 29, 24, 31, 10, 20, 25, 35, 11, 18, 16, 9, 8,
         7, 4, 3, 6, 1]
    ],
    "neg_samples": [33, 19, 15, 27, 34, 14, 23, 26, 12, 32]
}

# 将neg_samples转换为集合提高查找效率
neg_samples_set = set(data["neg_samples"])

# 构建pos_neg_pairs：每个正样本对应其负样本索引的交集
pos_neg_pairs = []
for i, pos_ctx in enumerate(data["positive_ctxs"]):
    # 获取当前正样本对应的负样本索引列表
    neg_indices = data["negative_ctxs"][i]

    # 计算与neg_samples的交集
    intersection = [idx for idx in neg_indices if idx in neg_samples_set]

    pairs = [(i, inx) for inx in intersection]
    pos_neg_pairs = pos_neg_pairs + pairs

