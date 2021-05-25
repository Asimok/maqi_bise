from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm, trange

from sequential_sentence_selector_module.utils import convert_examples_to_features, DataProcessor, warmup_linear

try:
    from sequential_sentence_selector_module.modeling_sequential_sentence_selector import BertForSequentialSentenceSelector
except:
    from modeling_sequential_sentence_selector import BertForSequentialSentenceSelector

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_file_path",
                        type=str,
                        default=None,
                        required=True,
                        help="File path to training data")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_sent_num",
                        default=30,
                        type=int)
    # 支持事实
    parser.add_argument("--max_sf_num",
                        default=15,
                        type=int)
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam. (def: 5e-5)")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    cpu = torch.device('cpu')

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = DataProcessor()

    # Prepare model
    if args.bert_model != 'bert-large-uncased-whole-word-masking':
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case)

        model = BertForSequentialSentenceSelector.from_pretrained(args.bert_model,
                                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                      -1))
    else:
        model = BertForSequentialSentenceSelector.from_pretrained('bert-large-uncased',
                                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                      -1))
        from utils import get_bert_model_from_pytorch_transformers

        state_dict, vocab_file = get_bert_model_from_pytorch_transformers(
            args.bert_model)
        model.bert.load_state_dict(state_dict)
        tokenizer = BertTokenizer.from_pretrained(
            vocab_file, do_lower_case=args.do_lower_case)

        logger.info(
            'The {} model is successfully loaded!'.format(args.bert_model))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    POSITIVE = 1.0
    NEGATIVE = 0.0

    # Load training examples
    train_examples = None
    num_train_steps = None
    # 加载数据
    train_examples = processor.get_train_examples(args.train_file_path)
    train_features = convert_examples_to_features(
        train_examples, args.max_seq_length, args.max_sent_num, args.max_sf_num, tokenizer, train=True)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total,
                         max_grad_norm=1.0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_masks = torch.tensor(
        [f.input_masks for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in train_features], dtype=torch.long)
    all_output_masks = torch.tensor(
        [f.output_masks for f in train_features], dtype=torch.float)
    all_num_sents = torch.tensor(
        [f.num_sents for f in train_features], dtype=torch.long)
    all_num_sfs = torch.tensor(
        [f.num_sfs for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids,
                               all_input_masks,
                               all_segment_ids,
                               all_target_ids,
                               all_output_masks,
                               all_num_sents,
                               all_num_sfs)
    # 训练数据抽样
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    epc = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            input_masks = batch[1]
            batch_max_len = input_masks.sum(dim=2).max().item()

            target_ids = batch[3]

            num_sents = batch[5]
            batch_max_sent_num = num_sents.max().item()

            num_sfs = batch[6]
            batch_max_sf_num = num_sfs.max().item()

            output_masks_cpu = (batch[4])[
                               :, :batch_max_sf_num + 1, :batch_max_sent_num + 1]

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, _, output_masks, __, ___ = batch
            B = input_ids.size(0)

            input_ids = input_ids[:, :batch_max_sent_num, :batch_max_len]
            input_masks = input_masks[:, :batch_max_sent_num, :batch_max_len]
            segment_ids = segment_ids[:, :batch_max_sent_num, :batch_max_len]
            target_ids = target_ids[:, :batch_max_sf_num]
            # 1 for EOE
            output_masks = output_masks[:, :batch_max_sf_num + 1, :batch_max_sent_num + 1]

            target = torch.FloatTensor(output_masks.size()).fill_(
                NEGATIVE)  # (B, NUM_STEPS, |S|+1) <- 1 for EOE
            for i in range(B):
                output_masks[i, :num_sfs[i] + 1, -1] = 1.0  # for EOE
                target[i, num_sfs[i], -1].fill_(POSITIVE)

                for j in range(num_sfs[i].item()):
                    target[i, j, target_ids[i, j]].fill_(POSITIVE)
            target = target.to(device)

            loss = model(input_ids, segment_ids, input_masks,
                         output_masks, target, target_ids, batch_max_sf_num)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += B
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * \
                               warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            args.output_dir, "pytorch_model_" + str(epc + 1) + ".bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        epc += 1


if __name__ == "__main__":
    main()
