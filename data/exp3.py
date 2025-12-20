import sys

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



input_data_path1 = "./step6_output/step6_negative_samples_support_facts_filterd.json"

data = load_json(input_data_path1)
co = int(len(data)/2)
data1 = data[:co]
data2 = data[co:]


with open("./step6_output/step6_negative_samples_support_facts_filterd_1.json", 'w', encoding='utf-8') as f:
    json.dump(data1, f, indent=2, ensure_ascii=False)

with open("./step6_output/step6_negative_samples_support_facts_filterd_2json", 'w', encoding='utf-8') as f:
    json.dump(data2, f, indent=2, ensure_ascii=False)

