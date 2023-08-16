import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer, util

class HybridEmbedder:
    def __init__(self, model_ID, device):
        # Save the model to device
        self.image_model = CLIPModel.from_pretrained(model_ID).to(device)
        # Get the processor
        self.image_processor = CLIPProcessor.from_pretrained(model_ID)
        # Get the tokenizer
        self.image_tokenizer = CLIPTokenizer.from_pretrained(model_ID)
        self.text_embedder = SentenceTransformer("all-mpnet-base-v2")
        self.pretty_name = "HYBRID"
        self.device = device
    
    def get_name (self,):
        return self.pretty_name
    
    def get_text_embedding(self, text, emb_type="text"):
        if emb_type == "text":
            text_embeddings = self.text_embedder.encode(text, convert_to_tensor=True)
        else:
            inputs = self.image_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            text_embeddings = self.image_model.get_text_features(**inputs)
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np

    def get_img_embedding(self, img):
        image = self.image_processor(text=None, images=img, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        embedding = self.image_model.get_image_features(image)
        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np