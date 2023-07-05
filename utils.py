import json


def load_jsonl_file(filename):
    with open(filename, "r", encoding='UTF-8') as json_file:
        json_list = list(json_file)
    data = []
    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    return data