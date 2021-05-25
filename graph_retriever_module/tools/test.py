import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm

from utils import DataProcessor, convert_examples_to_features


class TestConfig:
    def __init__(self):
        self.train_file_path = '../data/hotpot/graph_retriever_dataset/simple_train_dataset.json'
        self.task = 'hotpot_distractor'
        self.example_limit = None
        self.max_redundant_num = 3
        self.tfidf_limit = None
        self.max_select_num = 3
        self.use_redundant = False
        self.open = False


if __name__ == "__main__":
    train_file_path = '../data/hotpot/graph_retriever_dataset/simple_train_dataset.json'
    config = TestConfig()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    p = DataProcessor()
    exp = p.get_train_examples(config)

    train_features_ = convert_examples_to_features(
        exp, 378, 10, config, tokenizer, train=True)

    all_input_ids = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in train_features_], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features_], dtype=torch.long)
    all_output_masks = torch.tensor([f.output_masks for f in train_features_], dtype=torch.float)
    all_num_paragraphs = torch.tensor([f.num_paragraphs for f in train_features_], dtype=torch.long)
    all_num_steps = torch.tensor([f.num_steps for f in train_features_], dtype=torch.long)
    # 本轮训练数据
    train_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_output_masks,
                               all_num_paragraphs, all_num_steps)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)
    d =[]
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        d= batch[0]
        break
    print(d.max().item())


