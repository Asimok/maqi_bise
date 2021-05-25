# graph_retriever详解

## 数据集格式

- json

  ```json
  [{"question":"","context":{},"q_id":"","tagged_context":{},"all_linked_para_title_dic":{},"all_linked_paras_dic":{},"short_gold":[],"redundant_gold":[],"all_redundant_gold":[]},{},.....,{}]
  ```

  

## 数据预处理

将每条数据封装为标准格式

```
InputExample(guid=guid,
                                         q=question,
                                         c=context,
                                         para_dic=all_linked_paras_dic,
                                         s_g=short_gold,
                                         r_g=redundant_gold,
                                         all_r_g=all_redundant_gold,
                                         all_paras=all_paras)
```

将每条数据通过Bert预训练模型转换为特征：

将问题编码：

```
tokens_q = tokenize_question(example.question, tokenizer)

# 原始数据
What investment authority owns the shopping centre near John Lewis Reading?
# 编码后
['[CLS]', 'what', 'investment', 'authority', 'owns', 'the', 'shopping', 'centre', 'near', 'john', 'lewis', 'reading', '?', '[SEP]']
```

将每条数据封装为标准格式

```
InputFeatures(input_ids=input_ids,
                          input_masks=input_masks,
                          segment_ids=segment_ids,
                          output_masks=output_masks,
                          num_paragraphs=num_paragraphs,
                          num_steps=num_steps,
                          ex_index=ex_index))
```

# Tensor

tensor维度：[1,max_para_num,max_seq_length]