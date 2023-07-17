import os
import flamingo_model
from data_loaders import qa_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json
from tqdm import tqdm
import pandas as pd

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
        generate_output_mmqa(3, model, data_loader)

    if "WebQA" in run_on:
        data = qa_dataset.get_dataset(
            "WebQA", "val"
        )
        data_loader = DataLoader(data)
        generate_output_webqa(3, model, data_loader)


def generate_output_mmqa(beams, baseline, data):
    blank_image = Image.open("resources/1x1_#00000000.png")
    answers = {}
    df = {'qid':[],
               'Q':[],
               'A':[]
    }
    for x in tqdm(data, position=0, leave=True):
        ques = x[0][0]
        qid = x[1][0]
        question = "Q:" + ques + "A:"
        ans = baseline.generate_answer(beams, [blank_image], question)
        ans = ''.join(ans.splitlines())
        answers[qid] = ans
        df['qid'].append(qid)
        df['Q'].append(ques)
        df['A'].append(ans)

    df = pd.DataFrame(df)
    df.to_csv("mmqa_base_dev.csv")

    with open("mmqa_base_dev_.json", "w") as outfile:
        json.dump(answers, outfile)
     

def generate_output_webqa(beams, baseline, data):
    blank_image = Image.open("resources/1x1_#00000000.png")
    df = {'qid':[],
               'Qcate':[],
               'Q':[],
               'A':[],
               'Keywords_A':[],
               'Output_conf':[],
               'Output':[]
    }
    count = 0 
    for x in tqdm(data, position=0, leave=True):
        ques = x[0][0]
        qid = x[1][0]
        question = "Q:" + ques + "A:"
        ans = baseline.generate_answer(beams, [blank_image], question)
        ans = ''.join(ans.splitlines())
        df['qid'].append(qid)
        df['Qcate'].append(x[3][0])
        df['Q'].append(ques)
        df['A'].append(x[2][0])
        df['Keywords_A'].append("TBD")
        df['Output_conf'].append(1)
        df['Output'].append(ans)

    df = pd.DataFrame(df)
    df.to_csv("webqa_base_dev.tsv", header=['Guid','Qcate','Q','A','Keywords_A','Output_conf','Output'])


if __name__ == "__main__":
    model = flamingo_model.FlamingoModel("anas-awadalla/mpt-1b-redpajama-200b", "anas-awadalla/mpt-1b-redpajama-200b", 1)
    if type(opts.data_set) == "str":
        opts.data_set = [opts.data_set]
    print(f"Running Experiment on {opts.data_set}")
    run_experiment(model, opts.data_set)
