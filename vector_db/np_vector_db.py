import numpy as np
import os
import pickle
from data_loaders.dataset_mmqa import MMQAKnowledgeBase
from data_loaders.dataset_webqa import WebQAKnowledgeBase
from sklearn.metrics.pairwise import cosine_similarity
from vector_db.vector_db import VectorDB
from tqdm import tqdm 

STORE = "db/"


class NumpySearch(VectorDB):
    def __init__(self, embedder, data_set="MMQA"):
        self.embedder = embedder
        self.meta_data = {}
        self.vectors = []
        self.idx = 0
        self.store_path = os.path.join(STORE, embedder.get_name())
        if os.path.exists(STORE):
            if os.path.exists(self.store_path):
                if os.path.exists(os.path.join(self.store_path, f"vectors_{data_set}.npy")):
                    self.vectors = np.load(os.path.join(self.store_path, f"vectors_{data_set}.npy"))
                    print(f"loaded {self.vectors.shape[0]} vectors")
                if os.path.exists(os.path.join(self.store_path, f"meta_data_{data_set}.pkl")):
                    with open(
                        os.path.join(self.store_path, f"meta_data_{data_set}.pkl"), "rb"
                    ) as handle:
                        self.meta_data = pickle.load(handle)
                        print("Loaded meta data from pickle file")

        if len(self.meta_data.keys()) == 0 or len(self.vectors) == 0:
            self.process_documents(data_set)

    def process_documents(self, data_set):
        self.meta_data = {}
        self.vectors = []
        print(f"Computing Embeddings for: {data_set}")
        if data_set == "WebQA":
            data = WebQAKnowledgeBase(
                "/data/users/sgarg6/capstone/webqa/data/WebQA_train_val.json",
                "/data/users/sgarg6/capstone/webqa/data/imgs.tsv"   
            )
        else:
            data = MMQAKnowledgeBase(
                "/data/users/sgarg6/capstone/multimodalqa/MMQA_texts.jsonl",
                "/data/users/sgarg6/capstone/multimodalqa/MMQA_images.jsonl",
                "/data/users/sgarg6/capstone/multimodalqa/final_dataset_images"
            )
        print(f"Computing text embeddings")
        for text in tqdm(data.get_all_texts()):
            embed = self.embedder.get_text_embedding(text["text"])
            self.vectors.append(embed)
            text["type"] = "text"
            self.meta_data[self.idx] = text
            self.idx += 1
        print(f"Computing image embeddings")
        for img in tqdm(data.get_all_images()):
            image_id = img["id"] if data_set == "WebQA" else img["path"]
            try:
                image = data.get_image(image_id)
                embed = self.embedder.get_img_embedding(image)
                self.vectors.append(embed)
                img["type"] = "img"
                self.meta_data[self.idx] = img
                self.idx += 1
            except Exception as e:
                print(e)
                print(img)
        self.vectors = np.stack(self.vectors)
        print(f"Storing embeddings")
        if not os.path.exists(STORE):
            os.mkdir(STORE)
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)
        np.save(os.path.join(self.store_path, f"vectors_{data_set}.npy"), self.vectors)
        with open(os.path.join(self.store_path, f"meta_data_{data_set}.pkl"), "wb") as handle:
            pickle.dump(self.meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def retrieve(self, text, result_type="hybrid", k=5):
        embed = self.embedder.get_text_embedding(text)
        distance_matrix = cosine_similarity(embed.reshape(1, -1), self.vectors)
        distances = distance_matrix[0]
        top_k_idx = np.argsort(distances)
        if result_type == "hybrid":
            top_k_docs = [self.meta_data[idx] for idx in top_k_idx[:k]]
        elif result_type == "text":
            top_k_docs = [
                self.meta_data[idx]
                for idx in top_k_idx[: 2 * k]
                if self.meta_data[idx]["type"] == "text"
            ][:k]
        else:
            top_k_docs = [
                self.meta_data[idx]
                for idx in top_k_idx[: 2 * k]
                if self.meta_data[idx]["type"] == "img"
            ][:k]

        return top_k_docs
