CUDA_VISIBLE_DEVICES=3 python3 run_reader_confidence.py \
--bert_model bert-base-uncased \
--output_dir outputs \
--train_file ../data/hotpot/reader_dataset/simple_hotpot_reader_train_data.json \
--predict_file ../data/hotpot/reader_dataset/complete/hotpot_dev_squad_v2.0_format.json \
--max_seq_length 384 \
--do_train \
--do_predict \
--do_lower_case \
--version_2_with_negative \
--train_batch_size 1 \
--num_train_epochs 2


CUDA_VISIBLE_DEVICES=0 nohup  python3 run_reader_confidence.py \
--bert_model bert-base-uncased \
--output_dir outputs_5 \
--train_file ../data/hotpot/reader_dataset/complete/hotpot_reader_train_data.json \
--predict_file ../data/hotpot/reader_dataset/complete/hotpot_dev_squad_v2.0_format.json \
--max_seq_length 384 \
--do_train \
--do_predict \
--do_lower_case \
--version_2_with_negative \
--train_batch_size 8 \
--num_train_epochs 5 >log_reader.txt 2>&1 &
