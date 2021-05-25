CUDA_VISIBLE_DEVICES=3 nohup  python eval_main.py \
--eval_file_path data/hotpot/eval/new_dataset/new_distractor_dev.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path hotpot_models/reader \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval >log_dis.txt 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup  python eval_main.py \
--eval_file_path data/hotpot/eval/new_dataset/new_fullwiki_dev.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path hotpot_models/reader \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval >log_full.txt 2>&1 &

python eval_main.py \
--eval_file_path data/hotpot/eval/hotpot_fullwiki_first_100.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path hotpot_models/reader \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval

CUDA_VISIBLE_DEVICES=0 nohup python eval_main.py \
--eval_file_path data/hotpot/eval/new_dataset/new_fullwiki_dev.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path hotpot_models/reader \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-base-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval >log_reader_full_3_base.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python eval_main.py \
--eval_file_path data/hotpot/eval/new_dataset/new_fullwiki_dev.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path reader_module/outputs \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval >log_reader_full_5.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python eval_main.py \
--eval_file_path data/hotpot/eval/new_dataset/new_fullwiki_dev.jsonl \
--eval_file_path_sp data/hotpot/eval/hotpot_dev_distractor_v1.json \
--graph_retriever_path hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path reader_module/outputs_5 \
--sequential_sentence_selector_path hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased \
--do_lower_case \
--tfidf_limit 500 \
--eval_batch_size 4 \
--pruning_by_links \
--beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 \
--max_para_num 2000 \
--sp_eval >log_reader_full_reader_5.txt 2>&1 &