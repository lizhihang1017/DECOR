

import sys
import os
import json

from utils.load_data import load_json, read_json_lines
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# 配置参数
model_name = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 文件路径
input_data_path = ""
output_data_path = ""
temp_output_path = ""  # 临时文件用于实时保存

# 创建输出目录
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)

# 加载输入数据
input_data = load_json(input_data_path)

# 检查是否有已处理的部分
processed_indices = set()
if os.path.exists(temp_output_path):
    print(f"检测到未完成的处理任务，将从中断处继续...")
    # 读取已处理的结果
    with open(temp_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_indices.add(data['index'])
            except:
                continue


# 实时保存处理结果
def save_partial_result(index, result):
    """将单个处理结果追加到临时文件"""
    with open(temp_output_path, 'a', encoding='utf-8') as f:
        # 添加索引以便恢复
        result_with_index = {"index": index, **result}
        f.write(json.dumps(result_with_index, ensure_ascii=False) + '\n')


# 最终保存完整结果
def save_final_results():
    """合并所有处理结果到最终文件"""
    results = []
    if os.path.exists(temp_output_path):
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 移除临时添加的索引
                    if "index" in data:
                        del data["index"]
                    results.append(data)
                except:
                    continue

    # 按原始顺序排序
    results.sort(key=lambda x: x.get("id", 0) if "id" in x else x.get("index", 0))

    # 保存到最终文件
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"完整结果已保存至 {output_data_path}")


# 处理主循环
try:
    for i in tqdm(
            range(0, len(input_data)),
            total=len(input_data),
            desc="生成命题化表示"
    ):
        # 跳过已处理的样本
        if i in processed_indices:
            continue

        item = input_data[i]
        ctxs = item["ctxs"]
        proposition = []

        for ctx in ctxs:
            title = ctx["title"]
            content = ctx["text"]
            input_text = f"Title: {title}. Content: {content}"
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids.to(device), max_new_tokens=512).cpu()
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                prop_list = json.loads(output_text)
            except json.JSONDecodeError:
                # 尝试修复常见的JSON格式问题
                try:
                    # 移除可能的额外字符
                    cleaned_text = output_text.strip()
                    if not cleaned_text.startswith('['):
                        cleaned_text = '[' + cleaned_text
                    if not cleaned_text.endswith(']'):
                        cleaned_text = cleaned_text + ']'
                    prop_list = json.loads(cleaned_text)
                except:
                    prop_list = []
                    print(f"[警告] 无法解析模型输出: {output_text[:100]}...")

            for prop in prop_list:
                proposition.append(prop)

        # 更新结果
        item["proposition"] = proposition

        # 实时保存当前处理结果
        save_partial_result(i, item)

        # 每处理10个样本打印一次进度
        if i % 10 == 0 and i > 0:
            print(f"已处理 {i + 1}/{len(input_data)} 个样本")

    # 所有处理完成后保存最终结果
    save_final_results()

except KeyboardInterrupt:
    print("\n处理被中断！已保存已完成的部分结果。")
    print(f"下次运行将从断点处继续处理")
    print(f"临时文件位置: {temp_output_path}")

except Exception as e:
    print(f"\n发生错误: {str(e)}")
    print("已保存已完成的部分结果。")
    print(f"临时文件位置: {temp_output_path}")
    raise e