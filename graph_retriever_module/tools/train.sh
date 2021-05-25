CUDA_VISIBLE_DEVICES=3  python3 run_graph_retriever.py \
--task hotpot_distractor \
--bert_model bert-base-uncased \
--do_lower_case \
--train_file_path ../data/hotpot/graph_retriever_dataset/simple_train_dataset.json \
--output_dir outputs1/ \
--max_para_num 10 \
--neg_chunk 8 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--max_select_num 3 \
--train_batch_size 4


CUDA_VISIBLE_DEVICES=3  python3 run_graph_retriever.py \
--task hotpot_distractor \
--bert_model bert-base-uncased \
--do_lower_case \
--train_file_path ../data/hotpot/graph_retriever_dataset/simple_train_dataset.json \
--output_dir outputs1/ \
--max_para_num 5 \
--neg_chunk 1 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--max_select_num 3 \
--train_batch_size 1