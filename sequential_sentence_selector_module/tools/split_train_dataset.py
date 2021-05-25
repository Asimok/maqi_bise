import json

# read_path = '../../data/hotpot/sequential_sentence_selector_dataset/complete/hotpot_sf_selector_order_train.json'
# out_path = '../../data/hotpot/sequential_sentence_selector_dataset/hotpot_sf_selector_order_train_simple.json'


read_path = '../../data/hotpot/sequential_sentence_selector_dataset/complete/hotpot_sf_selector_order_dev.json'
out_path = '../../data/hotpot/sequential_sentence_selector_dataset/hotpot_sf_selector_order_dev_simple.json'


jsn = json.load(open(read_path, 'r'))
jsn_1 = jsn[0:10]
# jsn_2 = {'data': jsn_1, 'version': 'v2.0'}

with open(out_path, 'w') as f_obj:
    json.dump(jsn_1, f_obj)


