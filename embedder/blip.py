import torch
from transformers import BlipModel, AutoTokenizer, AutoProcessor
import PIL.Image as Image
import numpy as np


class BLIPEmbedder:
    def __init__(self, check_point, device):
        self.device = device
        # Load the BLIP2 model
        self.model = BlipModel.from_pretrained(check_point)
        self.model.to(self.device)

        # Create a BlipProcessor instance
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self.processor = AutoProcessor.from_pretrained(check_point)

        
    def get_img_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)
        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np

    def get_text_embedding(self, text):
        
        inputs = self.processor(text=text, truncation=True, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)

        # convert the embeddings to numpy array
        embedding_as_np = text_features.cpu().detach().numpy()
        return embedding_as_np
    
if __name__ == "__main__":

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    check_point = "Salesforce/blip-image-captioning-base"
    blip = BLIPEmbedder(check_point, device)

    image_file =  r"/data/users/sgarg6/capstone/multimodalqa/final_dataset_images/000487a1a0f36efac49107ea9a495ef9.JPG"
    image = Image.open(image_file)
    embed = blip.get_img_embedding(image)

    print(f"Embedding shape: {embed.shape}")

    embed = blip.get_text_embedding("Capstone project rocks!")
    print(f"Text embedding shape: {embed.shape}")
    
    
