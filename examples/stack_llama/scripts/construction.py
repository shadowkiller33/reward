from LHS import LHS
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from simcse import SimCSE
from nltk.corpus import words
import nltk
# Downloading the set of English words from nltk
nltk.download('words')
nltk.download('punkt')
# Create a set of English words
english_words = set(words.words())


def is_english(s):
    # Tokenizing the string into words
    words_in_s = nltk.word_tokenize(s.lower())

    # Count the number of words that are in the English word set
    english_word_count = sum(1 for word in words_in_s if word in english_words)

    # Check the ratio of English words to the total number of words
    return english_word_count / len(words_in_s) > 0.5

task = 'sum'
model = SimCSE(model_name_or_path="princeton-nlp/sup-simcse-roberta-base")
#model = LHS(model_name_or_path="princeton-nlp/sup-simcse-roberta-base", threshold=0.025, LHS_num=5000,LSH_conduct=False, interval=False)
sentence_list = []
response_j = []
response_k = []
import json
with open('/tmp/trl_lingfengshen/examples/stack_llama/sum/data.json', 'r') as f:
    data = json.load(f)
    L = len(data)
    for x in data:
        # if is_english(x['question']) == False:
        #     continue
        #if x['question'] not in sentence_list:
        sentence_list.append(x['question'])
        response_k.append(x['response_k'])
        response_j.append(x['response_j'])
    # sentence_list = data['question']
    # response_k = data['response_k']
    # response_j = data['response_j']

# for x,y,z in zip(sentence_list_1, response_k_1, response_j_1):
#     if is_english(x) == False:
#         continue
#     sentence_list.append(x)
#     response_k.append(y)
#     response_j.append(z)
# eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train").select(range(1000))
#train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train").select(range(200000))
# from tqdm import tqdm
# for id, x in tqdm(enumerate(eval_dataset)):
#     if len(x['question'].split()) >= 50:
#         continue
#     if x['question'] not in sentence_list:
#         sentence_list.append(x['question'])
#         response_k.append(x['response_k'])
#         response_j.append(x['response_j'])
# print(len(sentence_list))
#
# sentence_list = list(set(sentence_list))


# dict = {'question':sentence_list, 'response_k':response_k, 'response_j':response_j}
# import json
# with open('eval_'+str(N)+'.json', 'w') as f:
#     json.dump(dict, f)
# print('Data saving done!')

texts = sentence_list#[:20]
N = len(texts)
print(N)
model.build_index(texts,batch_size=512, faiss_fast=False, load = False, save = False, load_path = 'embedding_'+str(task)+'_'+str(N)+'.pkl', save_path = 'embedding_'+str(task)+'_'+str(N)+'.pkl' )

#for x in sentence_list[:10]:
query = sentence_list#[:10000]
threshold = 0.8
upper = 0.9
print("begin search")
retrieval = model.search(query, top_k=2, threshold=threshold, upper=upper)
print("search done")

dataset = []

for x, y in zip(retrieval, query):
    if x == [] or len(x) < 1:
        continue
    content = x[-1][0]
    data = {}
    data['query'] = y
    data['retrieval'] = content
    query_index = sentence_list.index(y)
    data['query_response_k'] = response_k[query_index]
    data['query_response_j'] = response_j[query_index]
    retrieval_index = sentence_list.index(content)
    data['retrieval_response_k'] = response_k[retrieval_index]
    data['retrieval_response_j'] = response_j[retrieval_index]
    dataset.append(data)
    # dataset['query'].append(y)
    # dataset['retrieval'].append(content)
    # print(f'Input prompt: {y}')
    # print(f'Retrieval: {content}')

L = len(dataset)
# use UTF-8 encoding to avoid Chinese character error
# with open('stack_'+str(L)+'_retrieval.json', 'w', encoding='utf-8') as f:
print(L)

import json
with open(str(task)+'_'+str(L)+'_retrieval_'+str(threshold)+'.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

# if retrieval == []:
#     print('empty')
#     continue
# retrieval = retrieval[0][0]
# print(f'Input prompt: {x}')
# print(f'Retrieval: {retrieval}')
#print(retrieval)


