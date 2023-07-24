import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RedpajamaModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", trust_remote_code=True)
        self.model = self.model.to(device)

    def generate_answer(self, beams, imgs, prompt):
        print(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
              **inputs, max_new_tokens=30, return_dict_in_generate=True, early_stopping=True, top_k=0, temperature=0.75, num_return_sequences=8, top_p=0.9,num_beams=8

        )

        token = outputs.sequences[0, input_length:]
        output_str = self.tokenizer.decode(token)
        return output_str