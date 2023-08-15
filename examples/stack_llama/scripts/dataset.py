from LHS import LHS
from datasets import load_dataset
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from simcse import SimCSE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
fig, ax = plt.subplots(dpi=600)

#model = SimCSE(model_name_or_path="princeton-nlp/sup-simcse-bert-base-uncased")
model = LHS(model_name_or_path="princeton-nlp/sup-simcse-bert-base-uncased", threshold=0.025, LHS_num=10000, interval=True)
sentence_list = []
eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train").select(range(20000))
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train").select(range(20000))

for x in eval_dataset['response_j']:
    sentence_list.append(x)
for x in train_dataset['response_j']:
    sentence_list.append(x)
sentence_list = list(set(sentence_list))
model.build_index(sentence_list,batch_size=512, faiss_fast=True)

L = []
A = []
score = []
for text in sentence_list[:10]:
    list = []
    length = len(text.split())
    aug = naw.ContextualWordEmbsAug(
        model_path='roberta-base', aug_min=int(0.054*length)+1,action="substitute")

    aug2 = nac.RandomCharAug(aug_char_min=int(0.054*length)+1,action="insert")
    augmented_text = aug.augment(text)
    augmented_texts = aug2.augment(augmented_text, n=10)
    for a in augmented_texts:
        L.append(a)
        A.append(text)

count = 0
for perturb_text, ori_text in zip(L, A):
    retrieval = model.search(perturb_text, top_k=1, threshold=-100)
    if retrieval == []:
        continue
    if retrieval[0][0] == ori_text:
        count += 1
print(count/len(L))


# gap_ = []
# avg = []
# max = []
# for i, text in enumerate(L):
#     similarities = model.similarity([text], sentence_list[:500])
#     retrieval = similarities[0][i]
#     max.append(retrieval)
#     #max = np.max(similarities[0])
#     # gap = retrieval- max
#     # gap_.append(gap)
#     for j, score in enumerate(similarities[0]):
#         if j == i:
#             continue
#         avg.append(score)
#     retrieval = []
#
# bplot1 = ax.boxplot([max, avg],  # notch shape
#                      vert=True,  # vertical box alignment
#                      patch_artist=True,  # fill with color
#                      labels=['max','avg'])  # will be used to label x-ticks
# plt.show()
# a = 1
# results = model.search(L, top_k=2)
# for i, x in enumerate(results):
#     retrieval.append(x[0][0])
#     score.append(x[0][0]==A[i])
# print(sum(score)/len(score))



# list.append((text, augmented_text, results))
# score.append(text==results)
# L.append(list)
# print(sum(score)/len(score))
a = 1



