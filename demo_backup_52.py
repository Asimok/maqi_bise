import argparse
import json
import logging

import remi.gui as gui
import torch
from remi import start, App

from pipeline.graph_retriever import GraphRetriever
from pipeline.reader import Reader
from pipeline.sequential_sentence_selector import SequentialSentenceSelector
from pipeline.tfidf_retriever import TfidfRetriever

parser = argparse.ArgumentParser()
global odqa


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


class ModelPredict:
    def __init__(self, args) -> object:

        self.args = args

        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        # print(device)
        # print(self.args.db_path)
        # print(self.args.tfidf_path)
        # TF-IDF Retriever
        self.tfidf_retriever = TfidfRetriever(self.args.db_path, self.args.tfidf_path)

        # Graph Retriever
        self.graph_retriever = GraphRetriever(self.args, device)

        # Reader
        self.reader = Reader(self.args, device)

        # Supporting facts selector
        self.sequential_sentence_selector = SequentialSentenceSelector(self.args, device)

    def predict(self,
                questions: list):

        print('-- Retrieving paragraphs by TF-IDF...', flush=True)
        tfidf_retrieval_output = []
        for i in range(len(questions)):
            question = questions[i]
            tfidf_retrieval_output += self.tfidf_retriever.get_abstract_tfidf('DEMO_{}'.format(i), question, self.args)

        print('-- Running the graph-based recurrent retriever model...', flush=True)
        graph_retrieval_output = self.graph_retriever.predict(tfidf_retrieval_output, self.tfidf_retriever, self.args)

        print('-- Running the reader_module model...', flush=True)
        answer, title = self.reader.predict(graph_retrieval_output, self.args)

        reader_output = [{'q_id': s['q_id'],
                          'question': s['question'],
                          'answer': answer[s['q_id']],
                          'context': title[s['q_id']]} for s in graph_retrieval_output]

        if self.args.sequential_sentence_selector_path is not None:
            print('-- Running the supporting facts retriever...', flush=True)
            supporting_facts = self.sequential_sentence_selector.predict(reader_output, self.tfidf_retriever, self.args)
        else:
            supporting_facts = []

        return tfidf_retrieval_output, graph_retrieval_output, reader_output, supporting_facts


class MyApp(App):
    txt_question: gui.TextInput
    txt_answer: gui.TextInput
    txt_retrieval: gui.TextInput
    txt_facts: gui.TextInput
    btn_question: gui.Button
    btn_re_question: gui.Button
    title: gui.Label
    empty: gui.Label

    def main(self):

        print('模型加载成功!')

        main_container = gui.Container(
            width="100%", height="100%", style={'margin': '0px auto'})

        horizontalContainer_txt = gui.Container(width='100%', height="55%",
                                                layout_orientation=gui.Container.LAYOUT_HORIZONTAL,
                                                margin='0px', style={'display': 'block', 'overflow': 'auto'})
        horizontalContainer_btn = gui.Container(width='100%', height="20%",
                                                layout_orientation=gui.Container.LAYOUT_HORIZONTAL,
                                                margin='0px',
                                                style={'display': 'block', 'overflow': 'auto', 'float': 'right'})

        verticalContainer = gui.Container(width="100%", height="100%", margin='0px auto',
                                          style={'display': 'block', 'overflow': 'hidden'})
        self.title = gui.Label('基于复杂问题关联的阅读理解系统', width="100%", height="3%", margin='10px',
                               style={'text-align': 'center', 'font-size': '25px'})
        self.empty = gui.Label('', width="70%", height="3%", margin='10px')
        # Reader results
        self.txt_answer = gui.TextInput(width="98%", height="10%", margin="10px",
                                        style={'font-size': '17px', 'padding': '10px'})
        self.txt_answer.set_text('推理答案:')

        # Retrieval results
        self.txt_retrieval = gui.TextInput(width="48%", height="90%", margin="10px",
                                           style={'font-size': '17px', 'padding': '10px'})
        self.txt_retrieval.set_text('推理路径:')

        # Supporting facts
        self.txt_facts = gui.TextInput(width="47%", height="90%", margin="10px",
                                       style={'font-size': '17px', 'padding': '10px'})
        self.txt_facts.set_text('事实依据:')

        # 问题
        self.txt_question = gui.TextInput(width="98%", height="12%", margin="10px",
                                          style={'font-size': '17px', 'padding': '10px'})
        self.txt_question.set_text('请输入你的问题:')

        # 按钮
        # 重新提问
        self.btn_re_question = gui.Button('重新提问', width=200, height=30, margin='10px',
                                          style={'font-size': '17px', 'float': 'right'})
        # setting the listener for the onclick event of the button
        self.btn_re_question.onclick.do(self.on_button_pressed_re_question)
        # 提问
        self.btn_question = gui.Button('推理', width=200, height=30, margin='10px',
                                       style={'font-size': '17px', 'float': 'right'})
        self.btn_question.onclick.do(self.on_button_pressed_question)

        # returning the root widget

        horizontalContainer_txt.append([self.txt_retrieval, self.txt_facts])
        horizontalContainer_btn.append([self.empty, self.btn_re_question, self.btn_question])
        verticalContainer.append(
            [self.title, self.txt_answer, horizontalContainer_txt, self.txt_question, horizontalContainer_btn])
        main_container.append(verticalContainer)

        print('系统启动成功!')
        return main_container

    @staticmethod
    def on_text_area_change(widget, newValue):
        # self.lbl.set_text('Text Area value changed!')
        print(str(newValue))

    def on_button_pressed_question(self, widget):
        self.txt_answer.set_text('正在推理...')
        try:
            question = self.txt_question.get_text().replace('请输入你的问题:', '')
        except:
            question = self.txt_question.get_text()
        print(question)
        questions = question.strip().split('|||')
        print('questions')
        global odqa
        tfidf_retrieval_output, graph_retriever_output, reader_output, supporting_facts = odqa.predict(questions)

        if graph_retriever_output is None:
            print()
            print('Invalid question! "{}"'.format(questions))
            self.txt_answer.set_text('Invalid question! "{}"'.format(questions))
            print()

        retrieval_results = json.dumps(graph_retriever_output, indent=4)
        print()
        print('#### Retrieval results ####')
        print(retrieval_results)
        print()
        print(graph_retriever_output[0]['context'])
        self.txt_retrieval.set_text("推理路径: \n" + json.dumps(graph_retriever_output[0]['context'], indent=4))

        reader_result = json.dumps(reader_output, indent=4)
        print('#### Reader results ####')
        print(reader_result)
        print()
        print(reader_output[0]['answer'])
        self.txt_answer.set_text("推理答案: \n" + json.dumps(reader_output[0]['answer'], indent=4).replace('"', ''))

        if len(supporting_facts) > 0:
            supporting_facts_str = json.dumps(supporting_facts, indent=4)
            print('#### Supporting facts ####')
            print(supporting_facts_str)
            print()
            print(supporting_facts[0]['supporting facts'])
            self.txt_facts.set_text("事实依据: \n" + json.dumps(supporting_facts[0]['supporting facts'], indent=4))

    def on_button_pressed_re_question(self, widget):
        self.txt_answer.set_text('重新提问')
        self.txt_question.set_text('请输入你的问题:')


def main():
    # Required parameters
    parser.add_argument("--graph_retriever_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Graph retriever model path.")
    parser.add_argument("--reader_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Reader model path.")
    parser.add_argument("--tfidf_path",
                        default=None,
                        type=str,
                        required=True,
                        help="TF-IDF path.")
    parser.add_argument("--db_path",
                        default=None,
                        type=str,
                        required=True,
                        help="DB path.")

    # Other parameters
    parser.add_argument("--sequential_sentence_selector_path",
                        default=None,
                        type=str,
                        help="Supporting facts model path.")
    parser.add_argument("--max_sent_num",
                        default=30,
                        type=int)
    parser.add_argument("--max_sf_num",
                        default=15,
                        type=int)

    parser.add_argument("--bert_model_graph_retriever", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--bert_model_sequential_sentence_selector", default='bert-large-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--max_seq_length",
                        default=378,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--max_seq_length_sequential_sentence_selector",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # RNN graph retriever-specific parameters
    parser.add_argument("--max_para_num",
                        default=10,
                        type=int)

    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=5,
                        help="Eval batch size")

    parser.add_argument('--beam_graph_retriever',
                        type=int,
                        default=1,
                        help="Beam size for Graph Retriever")
    parser.add_argument('--beam_sequential_sentence_selector',
                        type=int,
                        default=1,
                        help="Beam size for Sequential Sentence Selector")

    parser.add_argument('--min_select_num',
                        type=int,
                        default=1,
                        help="Minimum number of selected paragraphs")
    parser.add_argument('--max_select_num',
                        type=int,
                        default=3,
                        help="Maximum number of selected paragraphs")
    parser.add_argument("--no_links",
                        action='store_true',
                        help="Whether to omit any links (or in other words, only use TF-IDF-based paragraphs)")
    parser.add_argument("--pruning_by_links",
                        action='store_true',
                        help="Whether to do pruning by links (and top 1)")
    parser.add_argument("--expand_links",
                        action='store_true',
                        help="Whether to expand links with paragraphs in the same article (for NQ)")
    parser.add_argument('--tfidf_limit',
                        type=int,
                        default=None,
                        help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")

    parser.add_argument("--split_chunk", default=100, type=int,
                        help="Chunk size for BERT encoding at inference time")
    parser.add_argument("--eval_chunk", default=500, type=int,
                        help="Chunk size for inference of graph_retriever_module")

    parser.add_argument("--tagme",
                        action='store_true',
                        help="Whether to use tagme at inference")
    parser.add_argument('--topk',
                        type=int,
                        default=2,
                        help="Whether to use how many paragraphs from the previous steps")

    parser.add_argument("--n_best_size", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    global odqa
    odqa = ModelPredict(parser.parse_args())
    start(MyApp, address='0.0.0.0', port=8082, start_browser=False, multiple_instance=True)


if __name__ == "__main__":
    with DisableLogger():
        main()

