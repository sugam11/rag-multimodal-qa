import pandas as pd
import re
import math

df = pd.read_csv("./openf_webqa_base_dev_new_fix.tsv", sep="\t")
df.drop(df.columns[0], axis=1)


answers = df.iloc[:,7]
for i, ans in enumerate(answers):
	if isinstance(ans, str):
		ans = ans.replace('"', '')
		ans = ans.replace('\t', '')
		ans = ans.replace('\n', '')
		ans = ans.replace('(', '')
		ans = ans.replace(')', '')
		ans = ans.replace(',', '')
		ans = ans.replace('\\', '')
		ans = ans.replace(' \', '')
		ans = ans.replace('\'', '')

	df.at[i,"Output"] = [ans]

answers = df.iloc[:,3]
for i, ans in enumerate(answers):
	if isinstance(ans, str):
		ans = ans.replace('"', '')
		ans = ans.replace('\t', '')
		ans = ans.replace('\n', '')
		ans = ans.replace('(', '')
		ans = ans.replace(')', '')
		ans = ans.replace(',', '')
		ans = ans.replace('\\', '')



	df.at[i,"Q"] = ans

answers = df.iloc[:,4]
for i, ans in enumerate(answers):
	if isinstance(ans, str):
		ans = ans.replace('"', '')
		ans = ans.replace('\t', '')
		ans = ans.replace('\n', '')
		ans = ans.replace('(', '')
		ans = ans.replace(')', '')
		ans = ans.replace(',', '')
		ans = ans.replace('\\', '')
		ans = ans.replace('\'', '')


	df.at[i,"A"] = [str(ans)]

#df['Output_conf'] = df['Output_conf'].astype(int, errors='ignore')
#answers = df.iloc[:,6].astype(int)
#for i, ans in enumerate(answers):
	#if not math.isnan(ans):
		#df.at[i,"Output_conf"] = int(ans)

df = df.drop(df.columns[0], axis=1)
df.to_csv("test.tsv", header=['Guid','Qcate','Q','A','Keywords_A','Output_conf','Output'], sep="\t", quotechar="\"")

