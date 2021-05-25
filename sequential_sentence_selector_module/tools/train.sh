python3 run_sequential_sentence_selector.py \
--bert_model bert-large-uncased-whole-word-masking \
--train_file_path ../data/hotpot/sequential_sentence_selector_dataset/hotpot_sf_selector_order_train_simple.json \
--output_dir outputs \
--do_lower_case \
--train_batch_size 12 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3 \
--learning_rate 3e-5