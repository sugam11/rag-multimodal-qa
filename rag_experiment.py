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
from embedder.blip import BLIPEmbedder

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
    path = "retrieval_image_text.csv"
    df.to_csv(path)

    path = "retrieval_image_text.json"
    with open(path, "w") as outfile:
         json.dump(answers, outfile)


def clean_output(output):
    output = output.split("Answer: ")[-1]
    output = re.sub('<[^>]+>', '', output)
    return output

def prompt_builder(q, text_docs):             
    context = ""
    if text_docs:
       doc_1 = text_docs[0]
       doc_2 = text_docs[1] 
       doc_3 = text_docs[2] 
       doc_4 = text_docs[3]
       doc_5 = text_docs[4]

    #prompt = <image> Question: Which Player(s), in One Day Internationals of Sawai Mansingh Stadium, is holding a trophy that has a golden sphere inside of it? Answer: Sachin Tendulkar <|endofchunk|> 
    prompt = "You are a helpful Question Answering assistant. You are being provided with a question along with an image and text documents to assist with answering the question. Use either the image or text document to answer the question. Follow the examples and answer the last question. <image> Text Document: " + doc_1 +  " Question: Which Title(s), in Filmography of Ben Piazza, has the left half of a woman's face on its poster? Answer: Tell Me That You Love Me, Junie Moon<|endofchunk|>" + " <image> Text Document: " + doc_2 +  " Question: What kind of horses race in the A.P. Warrior events with One and One-Quarter Miles? Answer: three-year-old Thoroughbreds<|endofchunk|>" + " <image> Text Document: " + doc_3 +  " Question: What color is Rachel Dratch wearing? Answer: black<|endofchunk|>" + " <image> Text Document: " + doc_4 +  " Question: who played mama on throw momma from the train Answer: Anne Ramsey<|endofchunk|>" + " <image> Text Document: " + doc_5 + " Question: " + q + " Answer: "
    #print(prompt)   
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
        #if ctr == 20:
           #break
        #ctr+=1
    #write_results(df, answers)


def evaluate_rag(data_loader, db, model, mmqa_retriever):
    blank_image = Image.open("resources/1x1_#00000000.png")
    answers = {}
    df = {'qid':[],
          'Q':[],
          'A':[],
          'Gold_A':[],
    }
    ctr = 0
    for x in tqdm(data_loader, position=0, leave=True):
        ques = x[0]
        qids = x[1]
        golds = x[2]
        prompts = []
        img_list = []
        mmqa_text = [text for text in db.get_all_texts()]
        mmqa_map = {text["id"]: text for text in mmqa_text}
        mmqa_img = [img for img in db.get_all_images()]
        mmqa_map_img = {img["id"]: img for img in mmqa_img}
        sample_img_ids = ["ddf8b52a8400deaf05940c5cad8169cd", "117d500aaa630023c4038b8268b309c0", "971b2305045ca074690333bcf928d841", "cb19a093ca8ac601a8228c343e66e40c"]
        sample_text_ids = ["7d567450c9e91b13727db7f9581a05ae", "cf7add03ba748689c02dd3bbdf7430c6", "46974e53316c95be39b43eeebcbc980d", "54ae8f9219afe262db5cce8d81b49463"]
        for i, q in enumerate(ques):
             gold = golds[i]
             df['Q'].append(q)
             df['Gold_A'].append(gold)

             txt_docs = []
             for txt in sample_text_ids:
                 txt_docs.append(mmqa_map[txt]['text'])

             texts_retrieved = mmqa_retriever.retrieve(q, "text")
             text_docs = [doc["text"] for doc in texts_retrieved[:1]]
             txt_docs.append(text_docs[0])

             img_docs = []
             for img in sample_img_ids:
                 img_docs.append(mmqa_map_img[img]['path'])

             imgs_retrieved = mmqa_retriever.retrieve(ques, "image", k=5000)[:1]
             for img in imgs_retrieved:
                 img_docs.append(img['path'])

             if img_docs:
                 imgs = [db.get_image(img) for img in img_docs]
                 img_list.append(model.process_imgs(imgs))
             else:
                 img_list.append(model.process_imgs([blank_image]))

             prompt = prompt_builder(q, txt_docs)
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
        #if ctr == 40:
          #break
        #ctr+=1
    write_results(df, answers)

evaluate_rag(data_loader, db, model, mmqa_retriever)
#evaluate_ground_truth(data_loader, db, model)