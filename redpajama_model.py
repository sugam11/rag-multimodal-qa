import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


class RedpajamaModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", torch_dtype=torch.float16)
        self.model = model.to('cuda:0')

    def generate_answer(self, prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = model.generate(
              **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
        )
        token = outputs.sequences[0, input_length:]
        output_str = tokenizer.decode(token)
        return output_str