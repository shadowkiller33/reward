from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import pandas as pd
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import ssl
import datasets
import OpenAttack
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
ssl._create_default_https_context = ssl._create_unverified_context


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=5,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
eval_dataset = eval_dataset.shuffle()

if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]



training_args = TrainingArguments(
    output_dir='output_name',
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)
# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, truncation=True, max_length=512, use_auth_token=True)
config = AutoConfig.from_pretrained(script_args.model_name)



if "llama" in script_args.model_name:
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    # required for gpt2
    tokenizer.pad_token = tokenizer.eos_token

def softmax(vector):
    e = np.exp(vector)
    return e / np.sum(e)
class RankWrapper(OpenAttack.classifiers.Classifier):
    def __init__(self, model: OpenAttack.classifiers.Classifier):
        self.model = AutoModelForSequenceClassification.from_pretrained(
     model, num_labels=1, torch_dtype=torch.float16
 ).cuda()

    def get_pred(self, inputs):
        pred = self.get_prob(inputs).argmax(axis=1)
        #prinf pred with a predix "pred:"
        # print(f"pred:{pred}")

        return pred

    def get_prob(self, input_):
        #response_j = input_#self.context.input["x"]
        reward_k = self.context.input["rewards"]
        # print(f"input:{type(input_)}")
        # print(f"input:{len(input_)}")

        ret = []
        for sent in input_:
            tokenized_j = tokenizer(sent, truncation=True, return_tensors='pt')
            # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
            reward_j = self.model(input_ids=tokenized_j["input_ids"].cuda(), attention_mask=tokenized_j["attention_mask"].cuda())[0].item()

            values = np.array([reward_j, reward_k])
            res = np.array(softmax(values))
            ret.append(res)
            #print(f"res:{res}")
        # tokenized_k = tokenizer(response_k, truncation=True, return_tensors='pt')
        # reward_j = self.model.get_prob(response_j)

        # print(rewards_j, rewards_k)

        # print(values)

        # res = np.array([res])
        result = np.array(ret)
        #print(f"result:{result}")
        #print("res:%s" % res)
        # print(res)
        return result#np.array(results)

victim = RankWrapper(script_args.model_name)
# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
victim.model.config.pad_token_id = tokenizer.eos_token_id
victim.model.config.use_cache = not script_args.gradient_checkpointing

def preprocess_function(examples):
    # tokenized_j = tokenizer("Question: " + examples["question"] + "\n\nAnswer: " + examples["response_j"], truncation=True)
    # tokenized_k = tokenizer("Question: " + examples["question"] + "\n\nAnswer: " + examples["response_k"], truncation=True)
    j = "Question: " + examples["question"] + "\n\nAnswer: " + examples["response_j"]
    k = "Question: " + examples["question"] + "\n\nAnswer: " + examples["response_k"]
    tokenized_k = tokenizer(k,max_length=512, truncation=True, return_tensors='pt')
    rewards_k = victim.model(input_ids=tokenized_k["input_ids"].cuda(), attention_mask=tokenized_k["attention_mask"].cuda())[
        0].item()
    # print(rewards_j, rewards_k)
    label = 0

    return {
        "x": j,
        "y": label,
        "answer_k": k,
        "rewards": rewards_k
    }
eval_dataset = eval_dataset.map(function=preprocess_function)


attacker = OpenAttack.attackers.PWWSAttacker()
attack_eval = OpenAttack.AttackEval(attacker, victim)
attack_eval.eval(eval_dataset, visualize=True)






# import json
# with open('/apdcephfs_cq2/share_1603164/data/lingfengshen/output/adv_output/stack_' + str(N) + '_data.json', 'w') as f:
#     json.dump(json_data, f)
