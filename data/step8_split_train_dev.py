import random
import json
import os
from sklearn.model_selection import train_test_split

from utils.load_data import load_json, read_json_lines

def split_dataset(data, train_ratio=0.8, shuffle=True, random_state=42):
    """
    将数据集划分为训练集和验证集

    参数:
    data (list): 完整数据集
    train_ratio (float): 训练集比例，默认为0.8
    shuffle (bool): 是否打乱数据，默认为True
    random_state (int): 随机种子，确保结果可复现

    返回:
    train_data (list): 训练集
    val_data (list): 验证集
    """
    # 确保输入是列表
    if not isinstance(data, list):
        raise ValueError("输入数据必须是列表类型")

    # 确保比例在0-1之间
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("训练集比例必须在0和1之间")

    # 计算划分点
    total_size = len(data)
    train_size = int(total_size * train_ratio)

    if shuffle:
        # 使用sklearn的train_test_split进行随机划分
        train_data, val_data = train_test_split(
            data,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )
    else:
        # 简单按顺序划分
        train_data = data[:train_size]
        val_data = data[train_size:]

    return train_data, val_data


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = 
    TRAIN_FILE = 
    VAL_FILE = 
    TRAIN_RATIO = 0.9  # 训练集比例
    RANDOM_STATE = 42  # 随机种子

    # 确保输出目录存在
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)

    # 1. 加载数据
    print(f"从 {INPUT_FILE} 加载数据...")
    full_data = load_json(INPUT_FILE)
    print(f"加载完成，共 {len(full_data)} 个样本")

    # 2. 划分数据集
    print(f"划分数据集：{TRAIN_RATIO * 100}% 训练集，{(1 - TRAIN_RATIO) * 100}% 验证集")
    train_data, val_data = split_dataset(
        full_data,
        train_ratio=TRAIN_RATIO,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print(f"训练集大小: {len(train_data)} 样本")
    print(f"验证集大小: {len(val_data)} 样本")

    # 3. 保存划分后的数据集
    print(f"保存训练集到 {TRAIN_FILE}")
    with open(TRAIN_FILE, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"保存验证集到 {VAL_FILE}")
    with open(VAL_FILE, "w") as f:
        json.dump(train_data, f, indent=2)

    print("数据集划分完成!")
