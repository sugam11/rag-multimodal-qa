from open_flamingo import create_model_and_transforms
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import torch
import dataset_mmqa
from PIL import Image

class FlamingoBaseline:
    def __init__(self):
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1
        )

    """
    Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape
     batch_size x num_media x num_frames x channels x height x width.
     In this case batch_size = 1, num_media = 3, num_frames = 1,
     channels = 3, height = 224, width = 224.
    """
    def process_imgs(self, imgs):
        vision_x = [self.image_processor(x).unsqueeze(0) for x in imgs]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        return vision_x

    """
    Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
     We also expect an <|endofchunk|> special token to indicate the end of the text
     portion associated with an image.
    """
    def process_text(self, txt):
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer([txt], return_tensors="pt",)
        return lang_x

    """
    Generate text
    """
    def generate_answer(self, imgs, txt):
        vision_x = self.process_imgs(imgs)
        lang_x = self.process_text(txt)
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=1,
        )
        print("Generated Answer: ", self.tokenizer.decode(generated_text[0]))


dev_data = dataset_mmqa.MMQAQuestionAnswerPairs("MMQA_dev.jsonl")
loader = DataLoader(dev_data)
baseline = FlamingoBaseline()

demo_image_one = Image.open('1x1_#FFFFFFFF.png')
for x in dev_data:
    baseline.generate_answer([demo_image_one], "Who is the US president?")