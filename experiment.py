import os

os.environ["TRANSFORMERS_CACHE"] = "/data/users/bfarrell/models/"

import flamingo_model
from data_loaders import qa_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json
from tqdm import tqdm

optparser = optparse.OptionParser()
optparser.add_option(
    "-d",
    "--data",
    dest="data_set",
    default=["MMQA", "WebQA"],
    help="MMQA or webQA data or both",
)
(opts, _) = optparser.parse_args()


def run_experiment(model, run_on=["MMQA", "WebQA"]):
    if "MMQA" in run_on:
        data = qa_dataset.get_dataset(
            "MMQA", "val"
        )
        data_loader = DataLoader(data)
        generate_output(1, model, data_loader)

    if "WebQA" in run_on:
        data = qa_dataset.get_dataset(
            "WebQA", "val"
        )
        data_loader = DataLoader(data)
        generate_output(1, model, data_loader)


def generate_output(beams, baseline, data):
    blank_image = Image.open("1x1_#00000000.png")
    answers = {}
    for x in tqdm(data, position=0, leave=True):
        ques = x[0][0]
        qid = x[1][0]
        ans = baseline.generate_answer(beams, [blank_image], ques)
        answers[qid] = ans

    path = opts.data_set + "_base_dev.json"
    with open(path, "w") as outfile:
        json.dump(answers, outfile)

    path = opts.data_set + "_base_dev_pretty.json"
    with open(path, "w", encoding="utf-8") as f:
        for k, v in answers.items():
            line = {"qid": k, "answer": " ".join(v.split())}
            f.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    model = flamingo_model.FlamingoModel("anas-awadalla/mpt-1b-redpajama-200b", "anas-awadalla/mpt-1b-redpajama-200b", 1)
    if type(opts.data_set) == "str":
        opts.data_set = [opts.data_set]
    print(f"Running Experiment on {opts.data_set}")
    run_experiment(model, opts.data_set)
