import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


class CLIPEmbedder:
    def __init__(self, model_ID, device):
        # Save the model to device
        self.model = CLIPModel.from_pretrained(model_ID).to(device)
        # Get the processor
        self.processor = CLIPProcessor.from_pretrained(model_ID)
        # Get the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_ID)
        self.pretty_name = "CLIP"
        self.device = device
    
    def get_name (self,):
        return self.pretty_name
    
    def get_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        # convert the embeddings to numpy array
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np

    def get_img_embedding(self, img):
        image = self.processor(text=None, images=img, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        embedding = self.model.get_image_features(image)
        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np


if __name__ == "__main__":
    # Set the device
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"
    # Get model, processor & tokenizer
    clip_model = CLIPEmbedder(model_ID, device)
    from PIL import Image

    # Test the model
    im = Image.open(
        r"/data/users/sgarg6/capstone/multimodalqa/final_dataset_images/000487a1a0f36efac49107ea9a495ef9.JPG"
    )
    x = clip_model.get_img_embedding(im)
    print(x.shape)
    print(x)
