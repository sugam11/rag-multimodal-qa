from data_loaders import dataset_mmqa
from data_loaders import dataset_webqa


def get_dataset(dataset="MMQA", split="train"):
    if dataset == "MMQA":
        if split == "train":
            filename = "/data/users/sgarg6/capstone/multimodalqa/MMQA_train.jsonl"
        elif split == "val":
            filename = "/data/users/sgarg6/capstone/multimodalqa/MMQA_dev.jsonl"

        return dataset_mmqa.MMQAQuestionAnswerPairs(filename)

    elif dataset == "WebQA":
        data = dataset_webqa.WebQAQuestionAnswer(
            "/data/users/sgarg6/capstone/webqa/data/WebQA_train_val.json"
        )
        if split == "train":
            return data.get_train_split()
        elif split == "val":
            return data.get_val_split()
