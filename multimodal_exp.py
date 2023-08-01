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

data = qa_dataset.get_dataset(
            "MMQA", "val"
        )
data_loader = DataLoader(data, batch_size=1)

from embedder.clip import CLIPEmbedder
device = "cuda" if torch.cuda.is_available() else "cpu"


db = dataset_mmqa.MMQAKnowledgeBase("/data/users/sgarg6/capstone/multimodalqa/MMQA_texts.jsonl", "/data/users/sgarg6/capstone/multimodalqa/MMQA_images.jsonl", "/data/users/sgarg6/capstone/multimodalqa/final_dataset_images")
texts = db.get_all_texts()
model = flamingo_model.FlamingoModel("anas-awadalla/mpt-1b-redpajama-200b", "anas-awadalla/mpt-1b-redpajama-200b", 1)
#for x in texts:
    #print(x)

blank_image = Image.open("resources/1x1_#00000000.png")
answers = {}
df = {'qid':[],
      'Q':[],
      'A':[],
      #'Gold_A':[],
      #'docs':[]
}
ctr = 0
for x in tqdm(data_loader, position=0, leave=True):
    #print(x)
    ques = x[0]
    qids = x[1]
    #golds = x[2]
    #print(golds)

    
    prompts = []
    imgs = []
    for i, q in enumerate(ques):
         #docs = x[3]
         #doc_ids = []
         #for doc in docs:
             #doc_ids.append(doc[i])
         #texts = db.get_all_texts()
         #docs_retrieved = []
         #for text in texts:
             #if text['id'] in doc_ids:
                #docs_retrieved.append(text['text'])

         #context = "".join([doc for doc in docs_retrieved[0:2]])   
         context = ""
         df['Q'].append(q)
         prompt = "<image> Passage: " + context + "\nAnswer the following question and output only the answer. \nQ " + q + "\nA: " 
         prompts.append(prompt)
         imgs.append(model.process_imgs([blank_image]))

    prompts = model.process_text(prompts)
    imgs = torch.stack(imgs, dim=0)
    out = model.generate_answer(8, imgs, prompts)
    for i, ans in enumerate(out):
        ans = re.sub('<image>[^>]+A:', '', ans)
        ans = re.sub('<[^>]+>', '', ans)
        #print("Answer: ", ans, " end of answer")

        qid = qids[i]
        #gold = golds[i]
        answers[qid] = ans
        df['qid'].append(qid)
        df['A'].append(ans)

        #df['Gold_A'].append(gold)
        #df['docs'].append(docs_retrieved)
    #if ctr == 2:
       #break
    #ctr += 1

#print(df)

df = pd.DataFrame(df)
path = "gold_mmqa_base_dev_2.csv"
df.to_csv(path)

path = "gold_mmqa_base_dev.json"
with open(path, "w") as outfile:
     json.dump(answers, outfile)

