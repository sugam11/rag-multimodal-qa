import flamingo_model
import dataset_mmqa
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import optparse
import json

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="data_set", default='MMQA', help="MMQA or webQA data")

data_set = 'MMQA'
if data_set == "MMQA":
	data = dataset_mmqa.MMQAQuestionAnswerPairs("MMQA_dev.jsonl")
	data_loader = DataLoader(data)
	baseline = flamingo_model.FlamingoBaseline()
	blank_image = Image.open('1x1_#00000000.png')

	answers = {}
	index = 0
	for x in data_loader:
		ques = x[0][0]
		qid = x[1][0]
		ans = baseline.generate_answer([blank_image], ques)
		answers[qid] = ans
		index = index + 1

	print(answers)
	with  open("MMQA_baseline_out.json", "w", encoding="utf-8") as f:
		for k, v in answers.items():
			line = {"qid": k, "answer": ' '.join(v.split())}
			f.write(json.dumps(line) + "\n")
	f.close()