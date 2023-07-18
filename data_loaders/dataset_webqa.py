# Author: Apala Thakur
# WebQA dataset interface
from torch.utils.data import Dataset
from data_loaders import data_utils
from PIL import Image
from io import BytesIO
import base64


class WebQAQuestionAnswer:
    def __init__(self, filename):
        self.train, self.val = data_utils.read_train_val(filename)
        self.train_dataset = WebQAQuestionAnswerPairs(self.train)
        self.val_dataset = WebQAQuestionAnswerPairs(self.val)

    def get_train_split(
        self,
    ):
        return self.train_dataset

    def get_val_split(
        self,
    ):
        return self.val_dataset


class WebQAQuestionAnswerPairs(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample["Q"], sample["Guid"], sample["A"]


class WebQAKnowledgeBase:
    def __init__(self, datafile, imgtsvfile):
        self.train, self.val = data_utils.read_train_val(datafile)
        self.imgDict={}
        with open(imgtsvfile, "r") as fp:
            lines = fp.readlines()
            
        for line in lines:
            id, img_base64 = line.strip().split("/t")
            self.imgDict[id] = img_base64
            
    def get_image(self, image_id):
        
        img_base64 = self.imgDict[image_id]  
        image = Image.open(BytesIO(base64.b64decode(img_base64))) 
        return image     
        
    def get_all_images(self):
        """
        {'title': '',
          'caption': '',
         'url': '',
         'id': '',
         'path': ''}
        """

        img = []
        for k, v in self.train:
            img += v["img_posFacts"] + v["img_negFacts"]

        for image in img:
            yield (
                {
                    "title": image["title"],
                    "caption": image["caption"],
                    "url": image["url"],
                    "id": image["image_id"],
                    "path": image["imageUrl"],
                }
            )

    def get_all_texts(self):
        """
        returns {'title': '',
         'url': '',
         'id': '',
         'text': ''}
        """
        txt = []
        for k, v in self.train:
            txt += v["txt_posFacts"] + v["txt_negFacts"]

        for text in txt:
            yield (
                {
                    "title": text["title"],
                    "url": text["url"],
                    "id": text["snippet_id"],
                    "text": text["fact"],
                }
            )
