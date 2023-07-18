import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RedpajamaModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-1b-redpajama-200b")
        self.model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True)
        self.model = self.model.to(device)

    def generate_answer(self, beams, imgs, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
              **inputs, max_new_tokens=128, do_sample=True, temperature=0, top_p=0.7, top_k=50, return_dict_in_generate=True
        )
        token = outputs.sequences[0, input_length:]
        output_str = self.tokenizer.decode(token)
        return output_str