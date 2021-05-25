import json
import pandas as pd

read_path = '../data/hotpot/db=wiki_hotpotqa.db_hotpotqa_new_test_tfidf_k=50.pruning_l=100_tag_me=False.prune_after_agg=False.prune_in_article=False_use_link=True_start=40000_end=50000.json'
out_path = '../data/hotpot/graph_retriever_dataset/simple_train_dataset.json'
# out_path2 = '../data/hotpot/graph_retriever_dataset/pd_dataset.json'
# read_data = pd.read_json(read_path)
# read_data_simple = read_data[0:10]
# pd.DataFrame(read_data_simple).to_json(out_path2)

r1 = '../../data/hotpot/graph_retriever_dataset/hotpot_train_order_sensitive.json'
o1 = '../../data/hotpot/graph_retriever_dataset/simple_hotpot_train_order_sensitive.json'
jsn = json.load(open(r1, 'r'))
jsn = jsn[0:10]

with open(o1, 'w') as f_obj:
    json.dump(jsn, f_obj)
