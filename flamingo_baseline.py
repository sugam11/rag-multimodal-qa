import os
os.environ['TRANSFORMERS_CACHE'] = '/data/users/bfarrell/models/'

import flamingo_model
import dataset_mmqa
import dataset_webqa
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json
from tqdm import tqdm

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="data_set", default='MMQA', help="MMQA or webQA data")
(opts, _) = optparser.parse_args()

def generate_output(baseline, data):
	blank_image = Image.open('1x1_#00000000.png')
	answers = {}
	for x in tqdm(data, position=0, leave=True):
		ques = x[0][0]
		qid = x[1][0]
		ans = baseline.generate_answer([blank_image], ques)
		answers[qid] = ans

	path = opts.data_set +"_base_dev.json"
	with open(path, "w") as outfile:
    		json.dump(answers, outfile)

	path = opts.data_set +"_base_dev_pretty.json"
	with  open(path, "w", encoding="utf-8") as f:
		for k, v in answers.items():
			line = {"qid": k, "answer": ' '.join(v.split())}
			f.write(json.dumps(line) + "\n")

if opts.data_set == "MMQA":
	data = dataset_mmqa.MMQAQuestionAnswerPairs("/data/users/sgarg6/capstone/multimodalqa/MMQA_dev.jsonl")
	data_loader = DataLoader(data)
	baseline = flamingo_model.FlamingoBaseline()

	generate_output(baseline, data_loader)

if opts.data_set == "webQA":
	webQA = dataset_webqa.WebQAQuestionAnswer("/data/users/sgarg6/capstone/webqa/data/WebQA_train_val.json")
	data = webQA.get_val_split()
	data_loader = DataLoader(data)
	baseline = flamingo_model.FlamingoBaseline()

	generate_output(baseline, data_loader)
