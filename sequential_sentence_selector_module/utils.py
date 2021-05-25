from __future__ import absolute_import, division, print_function

import json
import logging
import os

from pytorch_transformers import (BertConfig,
                                  BertModel, BertTokenizer)
from tqdm import tqdm

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_bert_model_from_pytorch_transformers(model_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    config = config_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, from_tf=bool('.ckpt' in model_name), config=config)

    tokenizer = tokenizer_class.from_pretrained(model_name)

    vocab_file_name = './vocabulary_' + model_name + '.txt'

    if not os.path.exists(vocab_file_name):
        index = 0
        with open(vocab_file_name, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    assert False
                    index = token_index
                writer.write(token + u'\n')
                index += 1

    return model.state_dict(), vocab_file_name


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, q, a, t, c, sf):
        self.guid = guid
        self.question = q
        self.answer = a
        self.titles = t
        self.context = c
        self.supporting_facts = sf


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, segment_ids, target_ids, output_masks, num_sents, num_sfs, ex_index):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.target_ids = target_ids
        self.output_masks = output_masks
        self.num_sents = num_sents
        self.num_sfs = num_sfs

        self.ex_index = ex_index


class DataProcessor:

    def get_train_examples(self, file_name):
        return self.create_examples(json.load(open(file_name, 'r')))

    def create_examples(self, jsn):
        examples = []
        max_sent_num = 0
        for data in jsn:
            guid = data['q_id']
            question = data['question']
            titles = data['titles']
            context = data['context']  # {title: [s1, s2, ...]}
            # {title: [index1, index2, ...]}
            supporting_facts = data['supporting_facts']

            max_sent_num = max(max_sent_num, sum(
                [len(context[title]) for title in context]))

            examples.append(InputExample(
                guid, question, data['answer'], titles, context, supporting_facts))

        return examples


def convert_examples_to_features(examples, max_seq_length, max_sent_num, max_sf_num, tokenizer, train=False):
    """Loads a data file into a list of `InputBatch`s."""

    DUMMY = [0] * max_seq_length
    DUMMY_ = [0.0] * max_sent_num
    features = []
    logger.info('#### Constructing features... ####')
    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):

        tokens_q = tokenizer.tokenize(
            'Q: {} A: {}'.format(example.question, example.answer))
        tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']

        input_ids = []
        input_masks = []
        segment_ids = []

        for title in example.titles:
            sents = example.context[title]
            for (i, s) in enumerate(sents):

                if len(input_ids) == max_sent_num:
                    break

                tokens_s = tokenizer.tokenize(
                    s)[:max_seq_length - len(tokens_q) - 1]
                tokens_s = tokens_s + ['[SEP]']

                padding = [0] * (max_seq_length -
                                 len(tokens_s) - len(tokens_q))

                input_ids_ = tokenizer.convert_tokens_to_ids(
                    tokens_q + tokens_s)
                input_masks_ = [1] * len(input_ids_)
                segment_ids_ = [0] * len(tokens_q) + [1] * len(tokens_s)

                input_ids_ += padding
                input_ids.append(input_ids_)

                input_masks_ += padding
                input_masks.append(input_masks_)

                segment_ids_ += padding
                segment_ids.append(segment_ids_)

                assert len(input_ids_) == max_seq_length
                assert len(input_masks_) == max_seq_length
                assert len(segment_ids_) == max_seq_length

        target_ids = []
        target_offset = 0

        for title in example.titles:
            sfs = example.supporting_facts[title]
            for i in sfs:
                if i < len(example.context[title]) and i + target_offset < len(input_ids):
                    target_ids.append(i + target_offset)
                else:
                    logger.warning('')
                    logger.warning('Invalid annotation: {}'.format(sfs))
                    logger.warning('Invalid annotation: {}'.format(
                        example.context[title]))

            target_offset += len(example.context[title])

        assert len(input_ids) <= max_sent_num
        assert len(target_ids) <= max_sf_num

        num_sents = len(input_ids)
        num_sfs = len(target_ids)

        output_masks = [([1.0] * len(input_ids) + [0.0] * (max_sent_num -
                                                           len(input_ids) + 1)) for _ in range(max_sent_num + 2)]

        if train:

            for i in range(len(target_ids)):
                for j in range(len(target_ids)):
                    if i == j:
                        continue

                    output_masks[i][target_ids[j]] = 0.0

            for i in range(len(output_masks)):
                if i >= num_sfs + 1:
                    for j in range(len(output_masks[i])):
                        output_masks[i][j] = 0.0

        else:
            for i in range(len(input_ids)):
                output_masks[i + 1][i] = 0.0

        target_ids += [0] * (max_sf_num - len(target_ids))

        padding = [DUMMY] * (max_sent_num - len(input_ids))
        input_ids += padding
        input_masks += padding
        segment_ids += padding

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_masks=input_masks,
                          segment_ids=segment_ids,
                          target_ids=target_ids,
                          output_masks=output_masks,
                          num_sents=num_sents,
                          num_sfs=num_sfs,
                          ex_index=ex_index))

    logger.info('Done!')

    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
