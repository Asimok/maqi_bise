import json

read_path = '../../data/hotpot/reader_dataset/complete/hotpot_reader_train_data.json'
out_path = '../../data/hotpot/reader_dataset/simple_hotpot_reader_train_data.json'

jsn = json.load(open(read_path, 'r'))
jsn_1 = jsn['data'][0:10]
jsn_2 = {'data': jsn_1, 'version': 'v2.0'}
with open(out_path, 'w') as f_obj:
    json.dump(jsn_2, f_obj)
