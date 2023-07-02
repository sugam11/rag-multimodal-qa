from torch.utils.data import Dataset


class MMQAQuestionAnswerPairs(Dataset):
    """
        Usage Sample: 
        dev_data = MMQADataset("/data/users/multimodalqa/MMQA_dev.jsonl")
    """
    
    def __init__(self, filename):
        self.data = utils.load_jsonl_file(filename)
        filtered_data = []
        for point in data:
            modalities = point["metadata"]["modalities"]
            if "table" not in modalities:
                filtered_data.append(point)
        self.data = filtered_data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point = self.data[idx]
        answers = [ans["answer"] for ans in point["answers"]]
        ques = point["question"]
        return ques, point["qid"], answers
    
    
    
class MMQAKnowledgeBase:
    """
        Usage Sample: 
        mmqa_kb = MMQAKnowledgeBase(
            "/data/users/multimodalqa/MMQA_texts.jsonl",
            "/data/users/multimodalqa/MMQA_images.jsonl",
        )
        for text in mmqa_kb.get_all_texts():
            print(text)
          
    """
    def __init__(self, text_kb_path, img_kb_path):
        self.text_kb = utils.load_jsonl_file(text_kb_path)
        print(f"Loaded {len(self.text_kb)} text passages")
        self.img_kb = utils.load_jsonl_file(img_kb_path)
        print(f"Loaded {len(self.img_kb)} image sources")

    def get_all_images(self):
        """
        Returns meta data of images in format:
        {'title': '',
         'url': '',
         'id': '',
         'path': ''}
        append the path key to actual image datastore to get the image
        """
        for img in self.img_kb:
            yield img

    def get_all_texts(self):
        """
        Returns meta data of images in format:
        {'title': '',
         'url': '',
         'id': '',
         'text': ''}
        """
        for text in self.text_kb:
            return text