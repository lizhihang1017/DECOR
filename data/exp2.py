import sys
import os

from utils.load_data import load_json, read_json_lines
from utils.eval_utils import single_ans_em
from utils.prompt import HotpotQA_WO_Doc_PROMPT, GET_POSITIVE_ANSWER,IR_COT, HotpotQA_PROMPT
from utils.eval_utils import single_ans_em
import transformers
import torch
import json
from tqdm import tqdm
import re
import string


input_data_path1 = "./step4_output/step4_4_positive_supporting_facts_filterd.json"
input_data_path2 = "./step6_output/step6_negative_samples_support_facts.jsonl"

data1 = load_json(input_data_path1)
dic_data1 = {t['id'] for t in data1}


data2 = list(read_json_lines(input_data_path2))

print(data2[0])

# new_data = []
# for item in tqdm(data2):
#     id = item['id']
#     if id in dic_data1:
#         new_data.append(item)

# with open("./step6_output/step6_negative_samples_support_facts_filterd.json", 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, indent=2, ensure_ascii=False)