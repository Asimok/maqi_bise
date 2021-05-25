import json

read_path = '../../data/hotpot/reader_dataset/complete/hotpot_reader_train_data.json'
read_path_dis = '../../data/hotpot/eval/hotpot_dev_distractor_v1.json'
read_path_full = '../../data/hotpot/eval/hotpot_dev_fullwiki_v1.json'

out_path = '../../data/hotpot/reader_dataset/simple_hotpot_reader_train_data.json'

jsn = json.load(open(read_path, 'r'))
jsn_dis = json.load(open(read_path_dis, 'r'))
jsn_full = json.load(open(read_path_full, 'r'))

jsn_1 = jsn['data'][0:10]
jsn_2 = {'data': jsn_1, 'version': 'v2.0'}
with open(out_path, 'w') as f_obj:
    json.dump(jsn_2, f_obj)
