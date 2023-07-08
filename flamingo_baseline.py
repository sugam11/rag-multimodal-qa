dev_data = dataset_mmqa.MMQAQuestionAnswerPairs("MMQA_dev.jsonl")
loader = DataLoader(dev_data)
baseline = FlamingoBaseline()

blank_image = Image.open('1x1_#00000000.png')
demo_qs = ['Who is the current president of the United States?', 'What does the president do?', 'What is 10 + 4?',
           'What colour is the sky']

answers = {}
index = 0
for x in demo_qs:
    ans = baseline.generate_answer([blank_image], x)
    answers[index] = ans
    index = index + 1

print(answers)