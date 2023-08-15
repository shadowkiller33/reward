from LHS import LHS
from datasets import load_dataset
import json
import nlpaug.augmenter.word as naw
import numpy as np

j_list = []
k_list = []
dataset = load_dataset('json', data_files='/tmp/trl_lingfengshen/examples/stack_llama/wmt/data.json', split='train')
eval_dataset = dataset.train_test_split(test_size=0.2)['test'].select(range(1000))
for x in eval_dataset['response_j']:
    j_list.append(x)
for x in eval_dataset['response_k']:
    k_list.append(x)

data = []
for i in range(len(j_list)):
    j_text = j_list[i]
    k_text = k_list[i]
    import nlpaug.augmenter.char as nac
    aug = nac.KeyboardAug()
    case = {}
    length = len(j_text.split())
    aug = naw.SpellingAug(aug_max=int(0.054*length), aug_min=int(0.054*length))
    augmented_texts = aug.augment(j_text, n=1)[0]
    case['response_j'] = augmented_texts
    case['response_k'] = k_text
    data.append(case)

with open('mt_typo.json', 'w') as outfile:
    json.dump(data, outfile)