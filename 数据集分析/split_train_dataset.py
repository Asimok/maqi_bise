import json
import pandas as pd

read_path = 'hotpot_train_v1.1.json'
out_path = 'simple_hotpot_train_v1.1.json'

jsn = json.load(open(read_path, 'r'))
jsn = jsn[0:10]

with open(out_path, 'w') as f_obj:
    json.dump(jsn, f_obj)
