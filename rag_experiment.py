import os
import flamingo_model
from data_loaders import qa_dataset, dataset_mmqa
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json
from tqdm import tqdm
import re
import pandas as pd
import torch
from embedder.clip import CLIPEmbedder


data = qa_dataset.get_dataset(
            "MMQA", "val"
        )

def collate_fn(x):
    ques = [sample[0] for sample in x]
    qids = [sample[1] for sample in x]
    answers = [sample[2] for sample in x]
    doc_ids = [sample[3] for sample in x]
    return ques, qids, answers, doc_ids

data_loader = DataLoader(data, batch_size = 1, collate_fn=collate_fn)
db = dataset_mmqa.MMQAKnowledgeBase("/data/users/sgarg6/capstone/multimodalqa/MMQA_texts.jsonl", "/data/users/sgarg6/capstone/multimodalqa/MMQA_images.jsonl", "/data/users/sgarg6/capstone/multimodalqa/final_dataset_images")
model = flamingo_model.FlamingoModel("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", "togethercomputer/RedPajama-INCITE-Instruct-3B-v1", 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
clip = CLIPEmbedder(model_ID, device)
from vector_db.np_vector_db import NumpySearch
mmqa_retriever = NumpySearch(clip, "MMQA") 


def write_results(df, answers):
    df = pd.DataFrame(df)
    path = "gold_mmqa_base_dev_rag.csv"
    df.to_csv(path)

    path = "gold_mmqa_base_rag.json"
    with open(path, "w") as outfile:
         json.dump(answers, outfile)


def clean_output(output):
    output = re.sub('<image>[^>]+A:', '', output)
    print(output)
    output = re.sub('<[^>]+>', '', output)
    return output


def prompt_builder(q, text_docs):             
    context = ""
    if text_docs:
       context = "".join([doc for doc in text_docs])  
    prompt = "<image> Passage: " + context + "\nAnswer the following question and output only the answer. \nQ " + q + "\nA: " 
    return prompt


def evaluate_ground_truth(data_loader, db, model):
    blank_image = Image.open("resources/1x1_#00000000.png")
    answers = {}
    df = {'qid':[],
          'Q':[],
          'A':[],
          'Gold_A':[],
          'docs':[]
    }
    ctr = 0
    for x in tqdm(data_loader, position=0, leave=True):
        ques = x[0]
        qids = x[1]
        golds = x[2]
        docs = x[3]   
        prompts = []
        img_list = []
        for i, q in enumerate(ques):
             gold = golds[i]
             doc_ids = docs[i]
             df['Q'].append(q)
             df['Gold_A'].append(gold)
             df['docs'].append(doc_ids)

             texts = db.get_all_texts()
             images = db.get_all_images()

             text_docs = []
             for text in texts:
                 if text['id'] in doc_ids:
                    text_docs.append(text['text'])

             img_docs = []
             for img in images:
                 if img['id'] in doc_ids:
                    img_docs.append(img["path"])

             if img_docs:
                 imgs = [db.get_image(img) for img in img_docs]
                 img_list.append(model.process_imgs(imgs))
             else:
                 img_list.append(model.process_imgs([blank_image]))

             prompt = prompt_builder(q, text_docs)
             prompts.append(prompt)

        prompts = model.process_text(prompts)
        imgs = torch.stack(img_list, dim=0)
        out = model.generate_answer(8, imgs, prompts)
        for i, ans in enumerate(out):
             ans = clean_output(ans)
             qid = qids[i]
             answers[qid] = ans
             df['qid'].append(qid)
             df['A'].append(ans)
        if ctr == 1:
           break
        ctr+=1
    write_results(df, answers)


def evaluate_rag(data_loader, db, model, mmqa_retriever):
    blank_image = Image.open("resources/1x1_#00000000.png")
    answers = {}
    df = {'qid':[],
          'Q':[],
          'A':[],
          'Gold_A':[],
          #'docs':[]
    }
    ctr = 0
    for x in tqdm(data_loader, position=0, leave=True):
        ques = x[0]
        qids = x[1]
        golds = x[2]
        prompts = []
        img_list = []
        for i, q in enumerate(ques):
             gold = golds[i]
             df['Q'].append(q)
             df['Gold_A'].append(gold)
             texts_retrieved = mmqa_retriever.retrieve(ques, "text")
             text_docs = [doc["text"] for doc in texts_retrieved[:1]]
             imgs_retrieved = mmqa_retriever.retrieve(ques, "image", k=5000)[:1]
             img_docs = [doc["path"] for doc in imgs_retrieved]
             if img_docs:
                 imgs = [db.get_image(img) for img in img_docs]
                 img_list.append(model.process_imgs(imgs))
             else:
                 img_list.append(model.process_imgs([blank_image]))

             prompt = prompt_builder(q, text_docs)
             print("prompt", prompt, "end")
             prompts.append(prompt)

        prompts = model.process_text(prompts)
        imgs = torch.stack(img_list, dim=0)
        out = model.generate_answer(8, imgs, prompts)
        for i, ans in enumerate(out):
             ans = clean_output(ans)
             qid = qids[i]
             answers[qid] = ans
             df['qid'].append(qid)
             df['A'].append(ans)
        if ctr == 10:
           break
        ctr+=1
    #write_results(df, answers)

evaluate_rag(data_loader, db, model, mmqa_retriever)
#evaluate_ground_truth(data_loader, db, model)