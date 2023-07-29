import os
import re
import flamingo_model
import redpajama_model
import llama_model
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


def run_experiment(model,f_prefix, run_on=["MMQA", "WebQA"]):
    if "MMQA" in run_on:
        data = qa_dataset.get_dataset(
            "MMQA", "val"
        )
        data_loader = DataLoader(data)
        generate_output_mmqa(10, model, data_loader, f_prefix)

    if "WebQA" in run_on:
        data = qa_dataset.get_dataset(
            "WebQA", "val"
        )
        #data_loader = DataLoader(data)
        #generate_output_webqa(10, model, data_loader, f_prefix)


def generate_output_mmqa(beams, baseline, data, f_prefix):
    blank_image = Image.open("resources/1x1_#00000000.png")
    answers = {}
    df = {'qid':[],
               'Q':[],
               'A':[]
    }
    for x in tqdm(data, position=0, leave=True):
        ques = x[0][0]
        qid = x[1][0]
        question =  "Please answer the following question and output only the correct answer \nQ: " + ques +" \nA: "
        ans = baseline.generate_answer(beams, [blank_image], question)
        ans = ''.join(ans.splitlines())
        ans = re.sub('<image>[^>]+ A:', '', ans)
        ans = re.sub('<[^>]+>', '', ans)
        #print(ques)
        #print(ans)
        answers[qid] = ans
        df['qid'].append(qid)
        df['Q'].append(ques)
        df['A'].append(ans)
    
    df = pd.DataFrame(df)
    path = f_prefix + "_mmqa_base_dev.csv"
    df.to_csv(path)

    path = f_prefix + "_mmqa_base_dev.json"
    with open(path, "w") as outfile:
        json.dump(answers, outfile)
     

def generate_output_webqa(beams, baseline, data, f_prefix):
    blank_image = Image.open("resources/1x1_#00000000.png")
    df = {'qid':[],
               'Qcate':[],
               'Q':[],
               'A':[],
               'Keywords_A':[],
               'Output_conf':[],
               'Output':[]
    }
    for x in tqdm(data, position=0, leave=True):
        ques = x[0][0]
        qid = x[1][0]
        question =  "Answer the following question and output only the correct answer. \nQ: " + ques +" \nA: "
        ans = baseline.generate_answer(beams, [blank_image], question)
        ans = ''.join(ans.splitlines())
        ans = re.sub('<image>[^>]+ A:', '', ans)
        ans = re.sub('<[^>]+>', '', ans)
        df['qid'].append(qid)
        df['Qcate'].append(x[3][0])
        df['Q'].append(ques)
        df['A'].append(x[2][0])
        df['Keywords_A'].append("TBD")
        df['Output_conf'].append(1)
        df['Output'].append(ans)

    df = pd.DataFrame(df)
    path = f_prefix + "_webqa_base_dev.tsv"
    df.to_csv(path, header=['Guid','Qcate','Q','A','Keywords_A','Output_conf','Output'], sep="\t")


if __name__ == "__main__":
    #model = flamingo_model.FlamingoModel("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", "togethercomputer/RedPajama-INCITE-Instruct-3B-v1", 1)
    if type(opts.data_set) == "str":
        opts.data_set = [opts.data_set]
    print(f"Running Experiment on {opts.data_set}")
    #run_experiment(model,"openf", opts.data_set)
    #model = redpajama_model.RedpajamaModel()
    #run_experiment(model,"redpj", opts.data_set)
    model = llama_model.Llama()
    run_experiment(model,"llama", opts.data_set)

