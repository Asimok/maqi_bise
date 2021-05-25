import json
import pandas as pd
read_path = 'hotpot_train_v1.1.json'
jsn = json.load(open(read_path, 'r'))
# 训练数据难度分布
# level_dict = {}
# for i in jsn:
#     if level_dict.get(i['level']):
#         level_dict[i['level']] += 1
#     else:
#         level_dict[i['level']] = 1
# level_desc = np.array(list(level_dict.values())) / np.array(list(level_dict.values())).sum()
# key_list = list(level_dict.keys())
# pd.DataFrame(key_list, list(level_desc)).to_excel('难度分布.xlsx')

# 段落数量
# context_len=[]
# for i in jsn :
#     context_len.append(len(i['context']))
# pd.Series(context_len).describe()

# 段落长度
# context_word_len = []
# distribution = {'50': 0, '100': 0, '150': 0, '200': 0, '250': 0, '300': 0, '350': 0, '400': 0, '450': 0, '>500': 0}
# for i in jsn:
#     temp_len = len(i['context'])
#     for j in range(temp_len):
#         m = len(i['context'][j][1][0])
#         context_word_len.append(m)
#         if m < 50:
#             distribution['50'] += 1
#         elif m < 100:
#             distribution['100'] += 1
#         elif m < 150:
#             distribution['150'] += 1
#         elif m < 200:
#             distribution['200'] += 1
#         elif m < 250:
#             distribution['250'] += 1
#         elif m < 300:
#             distribution['300'] += 1
#         elif m < 350:
#             distribution['350'] += 1
#         elif m < 400:
#             distribution['400'] += 1
#         elif m < 450:
#             distribution['450'] += 1
#         else:
#             distribution['>500'] += 1
# pd.Series(context_word_len).describe()
# pd.DataFrame(distribution.keys(), distribution.values()).to_excel('段落长度.xlsx')

# 支撑事实文本长度
supporting_facts_len = []
for i in jsn:
    m = len(i['supporting_facts'])
    supporting_facts_len.append(m)
pd.Series(supporting_facts_len).describe()