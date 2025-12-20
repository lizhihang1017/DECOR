

import sys
import os

from utils.load_data import load_json, read_json_lines
from utils.prompt import HotpotQA_WO_Doc_PROMPT, GET_POSITIVE_ANSWER,IR_COT
from utils.eval_utils import single_ans_em
import transformers
import torch
import json
from tqdm import tqdm
import re

input_data_path = 
output_data_path = 
model_name_or_path =

# 确保输出目录存在
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

# 加载输入数据
data = load_json(input_data_path)

# 初始化模型
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name_or_path,
    tokenizer=tokenizer,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "pad_token_id": tokenizer.eos_token_id
    },
    device_map="auto",
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 加载已处理的样本ID
processed_ids = set()
if os.path.exists(output_data_path):
    with open(output_data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                processed_ids.add(record["id"])
            except json.JSONDecodeError:
                continue  # 跳过无效行

# 打开输出文件以追加模式
output_file = open(output_data_path, "a", encoding="utf-8")

# 处理每个样本
for i in tqdm(range(len(data)), desc="step4 getting positive samples"):
    sample = data[i]
    sample_id = sample["id"]

    # 跳过已处理的样本
    if sample_id in processed_ids:
        continue

    question = sample["question"]

    # 构建命题字符串
    pro_str = ""
    pro_list = sample["proposition"]
    for j, prop in enumerate(pro_list):
        pro_str += f"[{j}] {prop}\n"

    prompt = GET_POSITIVE_ANSWER % (question, pro_str) # 回答引用

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": prompt},
    ]

    try:
        # 生成模型响应
        outputs = pipeline(
            messages,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][-1]["content"]

        # 解析响应
        if "No answer" in response:
            selected_idxs = []
        else:
            matches = re.findall(r'\[([\d, ]+)\]', response)
            try:
                selected_idxs = set()
                for match in matches:
                    for num in match.split(','):
                        num = num.strip()
                        if num.isdigit():
                            selected_idxs.add(int(num))
            except Exception as e:
                print(f"Error parsing indexes for sample {sample_id}: {e}")
                selected_idxs = []
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        response = ""
        selected_idxs = []

    # 创建结果对象
    result = {
        "id": sample_id,
        "question": question,
        "positive_ctxs": list(selected_idxs),
        "model_response": response  # 保存原始响应用于调试
    }

    # 添加到结果集并更新已处理ID
    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    output_file.flush()  # 确保立即写入磁盘
    os.fsync(output_file.fileno())  # 强制刷新文件缓冲区
    processed_ids.add(sample_id)

# 关闭文件
output_file.close()
print("Processing completed!")