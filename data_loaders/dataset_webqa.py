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
        return sample["Q"], sample["Guid"], sample["A"], sample['Qcate'], sample['Keywords_A']


class WebQAKnowledgeBase:
    def __init__(self, datafile, imgtsvfile):
        self.train, self.val = data_utils.read_train_val(datafile)
        self.imgDict={}
        with open(imgtsvfile, "r") as fp:
            lines = fp.readlines()
            
        for line in lines:
            img_id, img_base64 = tuple(line.strip().split("\t"))
            self.imgDict[int(img_id)] = img_base64
            
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

        img_list = []
        img_id = set()
        for point in self.train:
            imgs, img_ids = self.get_unique_from_list(img_id, point["img_posFacts"], "image_id")
            img_list += imgs
            img_id.update(img_ids)
            imgs, img_ids = self.get_unique_from_list(img_id, point["img_negFacts"], "image_id") 
            img_list += imgs
            img_id.update(img_ids)
        print(f"Fetching {len(img_list)} images")
        for image in img_list:
            yield (
                {
                    "title": image["title"],
                    "caption": image["caption"],
                    "url": image["url"],
                    "id": image["image_id"],
                    "path": image["imgUrl"],
                }
            )
    
    def get_unique_from_list(self, set_id, list_dic, item_key):
        out = []
        unique_ids = set()
        for item in list_dic:
            if item[item_key] not in set_id:
                unique_ids.add(item[item_key])
                out.append(item)
        return out, set_id

    def get_all_texts(self):
        """
        returns {'title': '',
         'url': '',
         'id': '',
         'text': ''}
        """
        text_list = []
        text_id = set()
        for point in self.train:
            texts, text_ids = self.get_unique_from_list(text_id, point["txt_posFacts"], "snippet_id")
            text_list += texts
            text_id.update(text_ids)
            texts, text_ids = self.get_unique_from_list(text_id, point["txt_negFacts"], "snippet_id") 
            text_list += texts
            text_id.update(text_ids)
        print(f"Fetching {len(text_list)} passages")
        for text in text_list:
            yield (
                {
                    "title": text["title"],
                    "url": text["url"],
                    "id": text["snippet_id"],
                    "text": text["fact"],
                }
            )
