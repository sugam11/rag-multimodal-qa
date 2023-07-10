import os
os.environ['TRANSFORMERS_CACHE'] = '/data/users/bfarrell/models/'

import flamingo_model
import dataset_mmqa
import webqa_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json
from tqdm import tqdm

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="data_set", default='MMQA', help="MMQA or webQA data")
(opts, _) = optparser.parse_args()


if opts.data_set == "MMQA":
	data = dataset_mmqa.MMQAQuestionAnswerPairs("/data/users/sgarg6/capstone/multimodalqa/MMQA_dev.jsonl")
	data_loader = DataLoader(data)
	baseline = flamingo_model.FlamingoBaseline()
	blank_image = Image.open('1x1_#00000000.png')

	answers = {}
	for x in tqdm(data_loader, position=0, leave=True):
		ques = x[0][0]
		qid = x[1][0]
		ans = baseline.generate_answer([blank_image], ques)
		answers[qid] = ans

	with  open("MMQA_baseline_out_test.json", "w", encoding="utf-8") as f:
		for k, v in answers.items():
			line = {"qid": k, "answer": ' '.join(v.split())}
			f.write(json.dumps(line) + "\n")
	f.close()


if opts.data_set == "webQA":
	data = webqa_dataset.WebQAQuestionAnswerPairs("/data/users/sgarg6/capstone/webqa/data/WebQA_test.json")
	data_loader = DataLoader(data)
	baseline = flamingo_model.FlamingoBaseline()
	blank_image = Image.open('1x1_#00000000.png')

	answers = {}
	for x in tqdm(data_loader):
		ques = x[0][0]
		qid = x[1][0]
		ans = baseline.generate_answer([blank_image], ques)
		answers[qid] = ans

	with  open("WebQA_baseline_out.json", "w", encoding="utf-8") as f:
		for k, v in answers.items():
			line = {"qid": k, "answer": ' '.join(v.split())}
			f.write(json.dumps(line) + "\n")
	f.close()
