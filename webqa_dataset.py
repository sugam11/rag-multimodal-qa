# Author: Apala Thakur
# WebQA dataset interface 
from torch.utils.data import Dataset
import webqa_utils

class WebQAQuestionAnswerPairs(Dataset):
    
    def __init__(self,filename):
        self.train, self.val = webqa_utils.read_train_val(filename)
    
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, index):
        data = self.train[index]
        return data['Q'], data['Guid'], data['A']
            
class WebQAKnowledgeBase: 
    
    def __init__(self,filename):
        self.train, self.val = webqa_utils.read_train_val(filename)
        
    def get_all_images(self):
        '''
        {'title': '',
          'caption': '',
         'url': '',
         'id': '',
         'path': ''}
        '''
        
        img = []
        for v in self.train:
            img += v['img_posFacts'] + v['img_negFacts']
        
        for image in img: 
            yield({'title': image['title'],
                   'caption': image['caption'],
                    'url': image['url'],
                    'id': image['image_id'],
                    'path': image['imageUrl']})
        
    def get_all_text(self):
        '''
        returns {'title': '',
         'url': '',
         'id': '',
         'text': ''}
        '''
        txt = []
        for v in self.train:
            txt += v["txt_posFacts"] + v["txt_negFacts"]
        
        for text in txt:
            yield({
                'title': text['title'],
                'url': text['url'],
                'id' : text['snippet_id'],
                'text': text['fact']
            }) 
            
          