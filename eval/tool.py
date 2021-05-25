import json

# 生成fullwiki setting数据集
import jsonlines

read_path = '../data/hotpot/eval/hotpot_dev_fullwiki_v1.json'
jsn = json.load(open(read_path, 'r'))
fullwiki_list = []
for item in jsn:
    ans = [item['answer']]
    fullwiki = {"id": item['_id'], "question": item['question'], "answer": ans}
    fullwiki_list.append(fullwiki)
# fullwiki_list=fullwiki_list[0:100]
with jsonlines.open('../data/hotpot/eval/new_dataset/new_fullwiki_dev.jsonl', 'w') as f:
    for item in fullwiki_list:
        # f.write(str(item).replace("'id'", '"id"').replace("'question'", '"question"').replace("'answer'", '"answer"'))
        f.write(item)
# 生成distractor setting
read_path = '../data/hotpot/eval/hotpot_dev_distractor_v1.json'
jsn = json.load(open(read_path, 'r'))
distractor_list = []
for item in jsn:
    ans = [item['answer']]
    distractor = {"id": item['_id'], "question": item['question'], "answer": ans}
    distractor_list.append(distractor)
distractor_list=distractor_list[0:100]
with jsonlines.open('../data/hotpot/eval/new_dataset/new_distractor_dev.jsonl', 'w') as f:
    for item in distractor_list:
        # f.write(str(item).replace("'id'", '"id"').replace("'question'", '"question"').replace("'answer'", '"answer"'))
        f.write(item)
