from LHS import LHS
from datasets import load_dataset
import json
from parrot import Parrot

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str):
    """
    Split the text into sentences.
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

j_list = []
k_list = []
dataset = load_dataset('json', data_files='/tmp/trl_lingfengshen/examples/stack_llama/sum/data.json', split='train')
eval_dataset = dataset.train_test_split(test_size=0.2)['test'].select(range(1000))
for x in eval_dataset['response_j']:
    j_list.append(x)
for x in eval_dataset['response_k']:
    k_list.append(x)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)
import nltk.data


data = []
from tqdm import tqdm
for i in tqdm(range(len(j_list))):
    j_text1 = j_list[i]
    k_text1 = k_list[i]

    case = {}
    split_j = split_into_sentences(j_text1)
    split_k = split_into_sentences(k_text1)
    j_text = split_j[0]
    k_text = split_k[0]
    j_para_phrases = parrot.augment(input_phrase=j_text,do_diverse=False, use_gpu=True,max_return_phrases = 1,adequacy_threshold = -0.05,
                                fluency_threshold = -0.05)[0][0]
    k_para_phrases = parrot.augment(input_phrase=k_text,do_diverse=False, use_gpu=True,max_return_phrases = 1,adequacy_threshold = -0.05,
                                fluency_threshold = -0.05)[0][0]
    split_j[0] = j_para_phrases
    split_k[0] = k_para_phrases
    j_para_phrases = ' '.join(split_j)
    k_para_phrases = ' '.join(split_k)

    case['response_j'] = j_para_phrases
    case['response_k'] = k_para_phrases
    data.append(case)

with open('sum_para.json', 'w') as outfile:
    json.dump(data, outfile)