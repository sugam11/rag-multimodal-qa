import json


def load_jsonl_file(filename):
    with open(filename, "r", encoding="UTF-8") as json_file:
        json_list = list(json_file)
    data = []
    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    return data


def read_train_val(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    train = []
    val = []

    for k, v in data.items():
        if v["split"] == "val":
            val.append(v)
        else:
            train.append(v)
    return train, val


def read_test(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    test = []
    for k, v in data.items():
        test.append(v)
    return test
