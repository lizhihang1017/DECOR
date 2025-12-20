"""
第一步，过滤掉靠大模型内部知识即可回答问题的数据，仅保留需要靠外部知识才可回答问题的样本。为后续银色标签即正样本生成做准备
"""



from utils.load_data import load_json,read_json_lines
from utils.prompt import HotpotQA_WO_Doc_PROMPT
from utils.eval_utils import single_ans_em
import transformers
import torch
import json
from tqdm import tqdm



input_data_path =
output_data_path =
model_name_or_path = 

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name_or_path,
    tokenizer=tokenizer,  # 使用配置好的tokenizer
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "pad_token_id": tokenizer.eos_token_id  # 显式设置pad_token_id
    },
    device_map="auto",
)
# 定义终止符列表
terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

data = load_json(input_data_path) # 原 hotpot_train_v1.1.json 文件
filtered_data = [] # 需靠外部知识才可正确回答的样本

for i in tqdm(
        range(0, len(data)),
        total=len(data),
        desc=f"step1 filtering"
):
    if i % 1000 == 0:
        print("当前大模型仅靠内部知识即可正确回答的样本 已有 {}".format(len(data) -  len(filtered_data)))
    question = data[i]["question"]
    gold_answer = data[i]["answer"]
    prompt = HotpotQA_WO_Doc_PROMPT.format(
        question=question
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id  # 确保pad_token_id设置正确
    )

    generated_answer = outputs[0]["generated_text"][-1]["content"]

    if single_ans_em(gold_answer, generated_answer) == 1: # 当前样本 LLM 仅靠内部知识即可回答问题，不符合要求
        continue

    filtered_data.append(data[i])

print("HotpotQA train set 共有 {}".format(len(data)))
print("当前大模型仅靠内部知识即可正确回答的样本 共有 {}".format(len(data) -  len(filtered_data)))
print("step 1 过滤出的样本共有 {}".format(len(filtered_data)))

with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
