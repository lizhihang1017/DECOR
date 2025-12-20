import json


def load_json(file_path):
    """加载检索结果文件"""
    if file_path.endswith('.jsonl'):
        # 处理JSON Lines格式
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        # 处理标准JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def read_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 解析每一行并将其添加到列表中
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data
