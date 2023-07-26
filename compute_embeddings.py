from embedder.clip import CLIPEmbedder
from vector_db.np_vector_db import NumpySearch
import torch


device = "cuda:5" if torch.cuda.is_available() else "cpu"

# Change the embedder here to clip/blip
model_ID = "openai/clip-vit-base-patch32"
embedder = CLIPEmbedder(model_ID, device)

# Change dataset name to MMQA or WebQA 
retriever = NumpySearch(embedder, "WebQA")