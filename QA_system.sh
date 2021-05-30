CUDA_VISIBLE_DEVICES=0 python QA_system.py \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path hotpot_models/reader \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--do_lower_case \
--max_para_num 200 \
--tfidf_limit 20 \
--pruning_by_links

#--reader_path /data0/maqi/maqi_bs/reader/outputs/


