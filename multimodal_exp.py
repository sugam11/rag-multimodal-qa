#!/usr/bin/env python
# coding: utf-8

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

data = qa_dataset.get_dataset(
            "MMQA", "val"
        )
data_loader = DataLoader(data)

from embedder.clip import CLIPEmbedder
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


db = dataset_mmqa.MMQAKnowledgeBase("/data/users/sgarg6/capstone/multimodalqa/MMQA_texts.jsonl", "/data/users/sgarg6/capstone/multimodalqa/MMQA_images.jsonl", "/data/users/sgarg6/capstone/multimodalqa/final_dataset_images")
texts = db.get_all_texts()
model = flamingo_model.FlamingoModel("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", "togethercomputer/RedPajama-INCITE-Instruct-3B-v1", 1)
#for x in texts:
    #print(x)

blank_image = Image.open("resources/1x1_#00000000.png")
answers = {}
ctr = 0
for x in tqdm(data_loader, position=0, leave=True):
    ques = x[0][0]
    docs = x[3]
    doc_ids = []
    for doc in docs:
        doc_ids.append(doc[0])
    texts = db.get_all_texts()
    docs_retrieved = []
    for text in texts:
        if text['id'] in doc_ids:
           docs_retrieved.append(text['text'])

    qid = x[1][0]
    context = "".join([doc for doc in docs_retrieved[0:1]])   
    prompt = "<image> Passage: " + context + "\nAnswer the following question and output only the answer. \nQ " + ques + "\nA: " 
    print("Prompt: ", prompt, " End of prompt")
    ans = model.generate_answer(8, [blank_image], prompt)
    ans = re.sub('<image>[^>]+A:', '', ans)
    ans = re.sub('<[^>]+>', '', ans)
    print("Answer: ", ans, " end of answer")
    #answers[qid] = ans
    ctr += 1
    if ctr == 10:
        break

