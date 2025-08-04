import torch
import json
import os
import argparse
from tqdm import tqdm
from persona_datasets import get_batch_iterator
import logging
from transformers import (
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    # RobertaForSequenceClassification,
    BertConfig,
    BertModel
)
import itertools
# import evaluate
import hydra
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from utils import get_local_run_dir
import string
import re
from collections import defaultdict
import numpy as np
from torch.nn import CrossEntropyLoss
from trainers import BasicTrainer
from tqdm import tqdm
import transformers
import torch.nn.functional as F
from utils import get_local_dir
import wandb
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
import re
from transformers.data.processors.utils import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import jieba
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.lora.request import LoRARequest


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        # self.roberta_single= BertConfig.from_config("/home/nfs03/ligr/model/upload/chinese-nli-cls/config.json")
        # self.roberta_single = BertModel(self.roberta_single)
        self.roberta_single = AutoModel.from_pretrained(
            "/home/nfs03/ligr/model/upload/chinese-nli-cls/")
        self.single_hidden2tag = RobertaClassificationHead(1024, tagset_size)

    def forward(self, input_ids, attention_mask):
        outputs_single = self.roberta_single(input_ids, attention_mask, None)
        # torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)
        hidden_states_single = outputs_single[1]

        score_single = self.single_hidden2tag(
            hidden_states_single)  # (batch, tag_set)
        return score_single


class RobertaClassificationHead(nn.Module):
    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features  # [:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def count_ngram(hyps_resp, n):
    """
    # Count the number of unique n-grams
    # :param hyps_resp: list, a list of responses
    # :param n: int, n-gram
    # :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


# hyps_resp = ["I am good", "you are nice",...]
def eval_distinct_avg(hyps_resp):
    """
    # compute distinct score for the hyps_resp
    # :param hyps_resp: list, a list of hyps responses
    # :return: average distinct score for 1, 2-gram
    """
    candidates = []
    for sentence in hyps_resp:
        for punctuation in string.punctuation:
            sentence = sentence.replace(
                punctuation, " {} ".format(punctuation))
        sentence = re.sub(" +", " ", sentence).strip()
        candidates.append(sentence.split(" "))
    num_tokens = sum([len(i) for i in candidates])
    dist1 = count_ngram(candidates, 1) / float(num_tokens)
    dist2 = count_ngram(candidates, 2) / float(num_tokens)

    return dist1, dist2, (dist1 + dist2) / 2.0


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    if type(generated) == list:
        generated_dict, reference_dict, index = {}, {}, 0
        for i, j in zip(generated, reference):
            generated_dict[index] = [i]
            reference_dict[index] = [j]
            index += 1
    score, scores = Bleu(4).compute_score(
        gts=generated_dict, res=reference_dict)
    # for idx, g in enumerate(generated):
    #     if is_corpus:
    #         score, scores = Bleu(4).compute_score(reference, {0: [g]})
    #     else:
    #         score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
    #                                               {0: [g]})
    # BLEUscore[0] = BLEUscore[0]/len(generated)
    # BLEUscore[1] = BLEUscore[1]/len(generated)
    # BLEUscore[2] = BLEUscore[2]/len(generated)
    # BLEUscore[3] = BLEUscore[3]/len(generated)
    rouge_score, rouge_scores = Rouge().compute_score(
        gts=generated_dict, res=reference_dict)
    return score, rouge_score, scores, rouge_scores


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for sentence in generated:
        for punctuation in string.punctuation:
            sentence = sentence.replace(
                punctuation, " {} ".format(punctuation))
        sentence = re.sub(" +", " ", sentence).strip()
        g = sentence.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
    return etp_score


def cal_ppl(test_iterator, model):
    ppl_metrics = []
    for batch in tqdm(test_iterator, desc="Calculate ppl"):
        input_batch = {}
        for k, v in batch.items():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                input_batch[k] = v.to(model.device)

        policy_chosen_logits = model(
            input_batch['chosen_input_ids'], attention_mask=input_batch['chosen_attention_mask']).logits.to(torch.float32)

        shift_logits = policy_chosen_logits[..., :-1, :].contiguous()
        shift_labels = input_batch['chosen_labels'][..., 1:].contiguous()
        # Flatten the tokens

        assert shift_logits.shape[:-1] == shift_labels.shape
        cross_entropy = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction='none')
        cross_entropy = cross_entropy.view(shift_labels.shape)
        label_size = torch.sum(shift_labels != -100,
                               dim=1).type_as(cross_entropy)

        # ppl = torch.exp(torch.mean(torch.sum(cross_entropy, dim=1)
        #                            .float()/label_size.float()))
        ppl = torch.sum(cross_entropy, dim=1).float()/label_size.float()

        ppl_metrics.extend(ppl.detach().cpu().numpy().tolist())
    ppl_metrics = np.exp(np.mean(ppl_metrics))
    return ppl_metrics


def generate(model_or_modelpath, tokenizer, config, strategy, device, generate_select=False, data_split="test", **kwargs):
    print("start generate, data ", data_split)
    model = LLM(model=model_or_modelpath, dtype='bfloat16', enable_lora=True, max_model_len = 10240)
    # sampling_params = BeamSearchParams(
    #     beam_width = config.num_beam,
    #     max_tokens = config.max_new_tokens
    # )
    sampling_params = SamplingParams(
        n =1,
        max_tokens = config.max_new_tokens,
    )
    

    # if config.is_chinese:
    #     eos_token_id = tokenizer.sep_token_id
    #     tokenizer.eos_token_id = tokenizer.sep_token_id
    # else:
    #     eos_token_id = tokenizer.eos_token_id

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_kwags = dict(
        mode=config.data_mode,
        single_turn=config.single_turn,
        with_prompt=config.with_prompt,
        only_persona_response=config.only_persona_response,
        small_data=config.is_small_data,
        is_chinese=config.is_chinese,
        is_llama3=config.is_llama3,
    )

    data_iterator_kwargs = dict(
        names=config.datasets,
        tokenizer=tokenizer,
        shuffle=False,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        sft_mode=True,
    )
    
    
    test_iterator = get_batch_iterator(**data_iterator_kwargs, split=data_split, n_epochs=1,
                                       batch_size=1, select_file_path=config.select_data_path, datasets_kwags=dataset_kwags)
    
    prompts, golden_response = [], []
    for batch in tqdm(test_iterator, desc="generating"):
        prompts.append(batch["prompt"][0])
        golden_response.append(batch["chosen_response_only"][0])
    
    lora_path = config.evaluate_model_path
    print(f"Loading Lora at {lora_path}")
    outputs = model.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("lora_adapter", 1, config.evaluate_model_path)
        )
    # outputs = model.beam_search(
    #     prompts,
    #     sampling_params,
    #     lora_request=LoRARequest("lora_adapter", 1, config.evaluate_model_path)
    # )
    
    model_generation = []
    for i, j, k in zip(prompts, golden_response, outputs):
        model_generation.append(dict(
            prompt=i,
            golden_response=j,
            generated_response=k.outputs[0].text,
        ))

    test_iterator_ppl = get_batch_iterator(**data_iterator_kwargs, split=data_split, n_epochs=1,
                                           batch_size=config.eval_batch_size, select_file_path=config.select_data_path, datasets_kwags=dataset_kwags)
    ppl = 0
    # ppl = cal_ppl(test_iterator_ppl, model)
    return model_generation, ppl


def cal_other_consistance(outputs_in: list, file_name: str):
    consistance_model_path = "/home/data_91_d/ligr/model/entailment_nli_model"
    idx, examples = 0, []
    len_pred = len(outputs_in)
    user_personas_count = []
    for item in outputs_in:
        idx = 0
        if type(item["pred"]) == str:
            pred = item["pred"]
        else:
            pred = item["pred"][0]
        if type(item["persona"]) == str:
            cnt = item["persona"]
            examples.append(InputExample(
                str(idx), cnt, pred, "0"))
        else:
            type(item["persona"]) == list
            for cnt in item["persona"]:
                examples.append(InputExample(
                    str(idx), cnt, pred, "0"))
                idx = idx+1
            user_personas_count.append(idx)
    print("Data prepared, total test data ", len_pred)

    tokenizer = AutoTokenizer.from_pretrained(
        consistance_model_path, use_fast=False)
    consistance_model = AutoModelForSequenceClassification.from_pretrained(
        consistance_model_path)
    print("model loaded")
    consistance_model.to("cuda")
    consistance_model.eval()

    def get_dataloader(input_examples, tokenizer, device, batch_size=256):
        features = convert_examples_to_features(
            input_examples,
            tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long).to(device)
        dataset = TensorDataset(all_input_ids, all_attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    train_dataloader = get_dataloader(
        examples, tokenizer, "cuda", batch_size=512)
    all_logits = None
    with torch.no_grad():
        # for batch in tqdm(train_dataloader):
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = consistance_model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].cpu().detach()
            else:  # [n, 3], 每个batch直接cat到第一个维度上
                all_logits = torch.cat(
                    (all_logits, outputs[0].cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)  # [n]
    # print (all_logits.shape, results.shape) # torch.Size([34971, 3]) torch.Size([34971])

    # calculate consistence score as https://aclanthology.org/P19-1542.pdf
    # label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
    assert sum(user_personas_count) == results.shape[0]
    cnt = 0
    results_idx = 0
    for per_user_count in user_personas_count:
        tmp_results = []
        for _ in range(per_user_count):
            tmp_results.append(results[results_idx] - 1)
            results_idx += 1
        cnt += max(tmp_results)
    # for i, res in enumerate(results):
    #     cnt = cnt + (res - 1)
    print('consistence score is ', cnt/len_pred)
    with open("/home/nfs01/ligr/persona_dialogue/consistance_score/"+file_name+str(cnt/len_pred)+".txt", 'w', encoding='utf8') as f:
        f.write(file_name+"\n")
        f.write("consistace score\n")
        f.write(str(cnt/len_pred))


def cal_ch_other_consistance(outputs_in: list, file_name: str):
    consistance_model_path = "/home/nfs03/ligr/model/upload/chinese-nli-cls/"
    idx, examples = 0, []
    len_pred = len(outputs_in)
    user_personas_count = []
    for item in outputs_in:
        idx = 0
        for cnt in item["persona"]:
            if type(item["pred"]) == str:
                pred = item["pred"]
            else:
                pred = item["pred"][0][0]
            examples.append(InputExample(
                str(idx), cnt, pred, "0"))
            idx = idx+1
        user_personas_count.append(idx)
    print(type(pred))
    print("Data prepared, total test data ", len_pred)

    tokenizer = AutoTokenizer.from_pretrained(
        consistance_model_path, use_fast=False)

    # 模型文件的路径
    model_path = os.path.join(consistance_model_path,
                              "Roberta_large_model_NLI.pt")
    consistance_model = RobertaForSequenceClassification(3)
    consistance_model.load_state_dict(torch.load(model_path))
    print("model loaded")
    consistance_model.to("cuda")
    consistance_model.eval()

    def get_dataloader(input_examples, tokenizer, device, batch_size=256):
        features = convert_examples_to_features(
            input_examples,
            tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long).to(device)
        dataset = TensorDataset(all_input_ids, all_attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    train_dataloader = get_dataloader(
        examples, tokenizer, "cuda", batch_size=256)
    all_logits = None
    with torch.no_grad():
        # for batch in tqdm(train_dataloader):
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = consistance_model(**inputs)
            if all_logits is None:
                all_logits = outputs.cpu().detach()
            else:  # [n, 3], 每个batch直接cat到第一个维度上
                all_logits = torch.cat(
                    (all_logits, outputs.cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)  # [n]
    # print (all_logits.shape, results.shape) # torch.Size([34971, 3]) torch.Size([34971])

    # calculate consistence score as https://aclanthology.org/P19-1542.pdf
    # label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
    assert sum(user_personas_count) == results.shape[0]
    results_idx = 0
    cnt = 0
    for per_user_count in user_personas_count:
        tmp_results = []
        for _ in range(per_user_count):
            tmp_results.append(results[results_idx] - 1)
            results_idx += 1
        cnt += max(tmp_results)
    # for i, res in enumerate(results):
    #     cnt = cnt + (res - 1)
    print('consistence score is ', cnt/len_pred)
    with open("/home/nfs01/ligr/persona_dialogue/consistance_score/"+file_name+str(cnt/len_pred)+".txt", 'w', encoding='utf8') as f:
        f.write(file_name+"\n")
        f.write("consistace score\n")
        f.write(str(cnt/len_pred))


def cal_consistance_score(generated_text: list, file_name: str):
    consistance_model_path = "/home/data_91_d/ligr/model/entailment_nli_model"
    idx, examples = 0, []
    len_pred = len(generated_text)
    user_personas_count = []
    print("start preprocess data")
    for item in generated_text:
        idx = 0
        if "prompt" not in item.keys():
            print(item)
            len_pred -= 1
        else:
            user_persona = re.search(
                ".*persona is described with:\"(.+)\".\n", item["prompt"]).group(1)
            per_persona = list(map(str.strip, user_persona.split(" .")))
            per_persona = [tmp_i[2:]
                           for tmp_i in per_persona if len(tmp_i) > 2]
            for cnt in per_persona:
                examples.append(InputExample(
                    str(idx), cnt, item["generated_response"], "0"))
                # examples.append(InputExample(
                #     str(idx), cnt, item["pred"], "0"))
                idx = idx+1
            user_personas_count.append(idx)
    print("Data prepared, total test data ", len_pred)

    tokenizer = AutoTokenizer.from_pretrained(
        consistance_model_path, use_fast=False)
    consistance_model = AutoModelForSequenceClassification.from_pretrained(
        consistance_model_path)
    print("model loaded")
    consistance_model.to("cuda")
    consistance_model.eval()

    def get_dataloader(input_examples, tokenizer, device, batch_size=256):
        features = convert_examples_to_features(
            input_examples,
            tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long).to(device)
        dataset = TensorDataset(all_input_ids, all_attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    train_dataloader = get_dataloader(
        examples, tokenizer, "cuda", batch_size=512)
    all_logits = None
    with torch.no_grad():
        # for batch in tqdm(train_dataloader):
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = consistance_model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].cpu().detach()
            else:  # [n, 3], 每个batch直接cat到第一个维度上
                all_logits = torch.cat(
                    (all_logits, outputs[0].cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)  # [n]
    # print (all_logits.shape, results.shape) # torch.Size([34971, 3]) torch.Size([34971])

    # calculate consistence score as https://aclanthology.org/P19-1542.pdf
    # label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
    # cnt = 0
    assert sum(user_personas_count) == results.shape[0]
    results_idx = 0
    cnt = 0
    for per_user_count in user_personas_count:
        tmp_results = []
        for _ in range(per_user_count):
            tmp_results.append(results[results_idx] - 1)
            results_idx += 1
        cnt += max(tmp_results)
    # for i, res in enumerate(results):
    #     cnt = cnt + (res - 1)
    print('consistence score is ', cnt/len_pred)
    with open("/home/nfs01/ligr/persona_dialogue/consistance_score/"+file_name+str(cnt/len_pred)+".txt", 'w', encoding='utf8') as f:
        f.write(file_name+"\n")
        f.write("consistace score\n")
        f.write(str(cnt/len_pred))


def cal_ch_consistance_score(generated_text: list, file_name: str):
    consistance_model_path = "/home/nfs03/ligr/model/upload/chinese-nli-cls"
    idx, examples = 0, []
    len_pred = len(generated_text)
    user_personas_count = []
    print("start preprocess data")
    for item in generated_text:
        idx = 0
        if "prompt" not in item.keys():
            print(item)
            len_pred -= 1
        else:
            user_persona = re.search(
                ".*用户的个性化信息为：\"(.+)\".\n", item["prompt"]).group(1)
            per_persona = list(map(str.strip, user_persona.split(" ")))
            per_persona = [tmp_i[2:]
                           for tmp_i in per_persona if len(tmp_i) > 2]
            for cnt in per_persona:
                examples.append(InputExample(
                    str(idx), cnt, item["pred"], "0"))
                # examples.append(InputExample(
                #     str(idx), cnt, item["generated_response"], "0"))

                # examples.append(InputExample(
                #     str(idx), cnt, item["pred"], "0"))
                idx = idx+1
            user_personas_count.append(idx)
    print("Data prepared, total test data ", len_pred)

    tokenizer = AutoTokenizer.from_pretrained(
        consistance_model_path, use_fast=False)

    # 模型文件的路径
    model_path = os.path.join(consistance_model_path,
                              "Roberta_large_model_NLI.pt")

    # 加载模型
    # consistance_model = torch.load(model_path)
    # print(type(consistance_model))
    consistance_model = RobertaForSequenceClassification(3)
    consistance_model.load_state_dict(torch.load(model_path))
    # consistance_model = AutoModelForSequenceClassification.from_pretrained(
    #     consistance_model_path)
    print("model loaded")
    consistance_model.to("cuda")
    consistance_model.eval()

    def get_dataloader(input_examples, tokenizer, device, batch_size=256):
        features = convert_examples_to_features(
            input_examples,
            tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long).to(device)
        dataset = TensorDataset(all_input_ids, all_attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    train_dataloader = get_dataloader(
        examples, tokenizer, "cuda", batch_size=512)
    all_logits = None
    with torch.no_grad():
        # for batch in tqdm(train_dataloader):
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = consistance_model(**inputs)
            if all_logits is None:
                all_logits = outputs.cpu().detach()
            else:  # [n, 3], 每个batch直接cat到第一个维度上
                all_logits = torch.cat(
                    (all_logits, outputs.cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)  # [n]
    # print (all_logits.shape, results.shape) # torch.Size([34971, 3]) torch.Size([34971])

    # calculate consistence score as https://aclanthology.org/P19-1542.pdf
    # label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
    assert sum(user_personas_count) == results.shape[0]
    results_idx = 0
    cnt = 0
    for per_user_count in user_personas_count:
        tmp_results = []
        for _ in range(per_user_count):
            tmp_results.append(results[results_idx] - 1)
            results_idx += 1
        cnt += max(tmp_results)
    # cnt = 0
    # for i, res in enumerate(results):
    #     cnt = cnt + (res - 1)
    print('consistence score is ', cnt/len_pred)
    with open("/home/nfs01/ligr/persona_dialogue/consistance_score/"+file_name+str(cnt/len_pred)+".txt", 'w', encoding='utf8') as f:
        f.write(file_name+"\n")
        f.write("consistace score\n")
        f.write(str(cnt/len_pred))


def evaluation(config: DictConfig):
    # rouge = evaluate.load(
    #     '/home/data_91_d/ligr/methods/evaluate/metrics/rouge')
    # bleu = evaluate.load('/home/data_91_d/ligr/methods/evaluate/metrics/bleu')
    # distinct = evaluate.load(
    #     "distinct")
    # entropy = evaluate.load(
    #     "/home/data_91_d/ligr/methods/evaluate/metrics/entropy")

    # model = transformers.AutoModelForCausalLM.from_pretrained(config.model.name_or_path)
    model = config.model.name_or_path

    # if config.evaluate_model_path != "null":
    #     print(
    #         f'loading pre-trained lora weights')
    #     model = PeftModel.from_pretrained(model, config.evaluate_model_path)
    #     model.merge_and_unload()
    #     print('loaded pre-trained weights')

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    generate_kwags = {}
    if "generate_select" not in config.keys():
        generate_kwags["generate_select"] = False
    else:
        generate_kwags["generate_select"] = True
    if "data_split" not in config.keys():
        generate_kwags["data_split"] = "test"
    else:
        generate_kwags["data_split"] = config.data_split
    # if "is_chinese" in config.keys() and config.is_chinese:
    #     generate_kwags["is_chinese"] = True
    # else:
    #     generate_kwags["is_chinese"] = False

    generated_text, ppl = generate(
        model, tokenizer, config, device="cuda", strategy="beam_search", **generate_kwags)

    goldens, generations = [], []
    for per_text in generated_text:
        goldens.append(per_text["golden_response"])
        generations.append(per_text["generated_response"])
    # rouge_score = rouge.compute(predictions=generations, references=goldens)
    if "generate_select" not in config.keys():
        save_dirs = os.path.join(get_local_dir(
            config.local_dirs), config.exp_name+".json")
    else:
        save_dirs = os.path.join(get_local_dir(
            config.local_dirs), config.exp_name+"_generate_select.json")
    with open(save_dirs, 'w', encoding='utf8') as f:
        json.dump(generated_text, f)

    if "is_chinese" in config.keys() and config.is_chinese:
        for i in range(len(goldens)):
            goldens[i] = " ".join(jieba.cut(goldens[i]))
            generations[i] = " ".join(jieba.cut(generations[i]))

    if "generate_select" not in config.keys():
        bleu_score, rouge_score, bleu_scores, rouge_scores = cal_BLEU_4(
            generated=generations, reference=goldens)

        dist1, dist2, mean_dist = eval_distinct_avg(generations)
        entropy = cal_entropy(generations)

        metric_dict = {
            "ppl": ppl,
            "dist_1": dist1,
            "dist_2": dist2,
            "dist_mean": mean_dist,
            "entropy_1": entropy[0],
            "entropy_2": entropy[1],
            "entropy_3": entropy[2],
            "entropy_4": entropy[3],
            "bleu_1": bleu_score[0],
            "bleu_2": bleu_score[1],
            "bleu_3": bleu_score[2],
            "bleu_4": bleu_score[3],
            "rouge_L": rouge_score
            # **rouge_score,
            # **bleu_score
        }

        bleu_scores = list(map(list, zip(*bleu_scores)))
        for item, bleu_s, rouge_s in zip(generated_text, bleu_scores, rouge_scores):
            item["bleu"] = bleu_s
            item["rouge_L"] = rouge_s

        wandb.log(metric_dict)

    else:
        generated_text.append({"ppl": ppl})
    print(f"Save at {save_dirs}")
    with open(save_dirs, 'w', encoding='utf8') as f:
        json.dump(generated_text, f)
    # print(metric_dict)

# todo C-score, it should use a RoBERTa model to calculate the C-score, we added it finally
# https://github.com/ChanLiang/PersonaClassifier/blob/main/cal_consistency_score.py


CONFIG_PATH = os.environ.get("CONFIG_PATH")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def run_generation(config: DictConfig):
    print(f"Load config from {CONFIG_PATH}")
    print(config)

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if "generate_select" not in config.keys():
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )
    evaluation(config)


if __name__ == "__main__":
    OmegaConf.register_new_resolver(
        "get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))
    run_generation()
