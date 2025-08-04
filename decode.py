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
        ppl = torch.sum(cross_entropy, dim=1).float()/label_size.float()

        ppl_metrics.extend(ppl.detach().cpu().numpy().tolist())
    ppl_metrics = np.exp(np.mean(ppl_metrics))
    return ppl_metrics


def generate(model_or_modelpath, tokenizer, config, strategy, device, generate_select=False, data_split="test", **kwargs):
    print("start generate, data ", data_split)
    if isinstance(model_or_modelpath, str):
        model = AutoModelForCausalLM.from_pretrained(model_or_modelpath)
    else:
        model = model_or_modelpath
    model.half()
    model.to(device)
    model.eval()

    if config.is_chinese:
        eos_token_id = tokenizer.sep_token_id
        tokenizer.eos_token_id = tokenizer.sep_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

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
    model_generation = []
    for batch in tqdm(test_iterator, desc="generating"):
        input_tokens = tokenizer(
            batch["prompt"][0], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones_like(input_tokens)
        if strategy == "beam_search":
            output_dic = model.generate(
                input_ids=input_tokens,
                attention_mask=attention_mask,
                num_beams=config.num_beam,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        else:
            # top10_top0.9_T0.9
            output_dic = model.generate(
                input_ids=input_tokens,
                topk=10,
                topp=0.9,
                do_sampling=False,
                temperature=1,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=eos_token_id,  # normal case
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
        all_reply = tokenizer.decode(
            output_dic['sequences'][0], skip_special_tokens=True)
        try:
            if config.is_chinese:
                if not generate_select:
                    generated_text = all_reply.replace(
                        " ", "").split("回复为：")[1]
                else:
                    generated_text = all_reply.replace(" ", "").split(
                        "最适当的个性化信息为：")[1]
            else:
                if not generate_select:
                    generated_text = all_reply.split("Response:")[1]
                else:
                    generated_text = all_reply.split(
                        "The preferred persona is:")[1]
        except:
            print(f"unexpected sentence: {all_reply}")
            generated_text = ""
        model_generation.append(dict(
            prompt=batch["prompt"][0],
            golden_response=batch["chosen_response_only"][0],
            generated_response=generated_text,
        ))
    test_iterator_ppl = get_batch_iterator(**data_iterator_kwargs, split=data_split, n_epochs=1,
                                           batch_size=config.eval_batch_size, select_file_path=config.select_data_path, datasets_kwags=dataset_kwags)
    ppl = 0
    return model_generation, ppl

def evaluation(config: DictConfig):
    print(config)
    policy_dtype = getattr(torch, config.model.policy_dtype)
    config.model.name_or_path = "/home/data_91_d/ligr/model/" + config.model.name_or_path
    print("Load from ", config.model.name_or_path)
    print("Datasets", config.datasets)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype)

    if config.evaluate_model_path != "null" and config.evaluate_model_path is not None:
        state_dict = torch.load(config.evaluate_model_path, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.evaluate_model_path} with metrics {json.dumps(metrics, indent=2)}')
        model.load_state_dict(state_dict['state'])
    else:
        print("No trained weights to load, use raw pre-trained model")

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

    generated_text, ppl = generate(
        model, tokenizer, config, device="cuda", strategy="beam_search", **generate_kwags)

    goldens, generations = [], []
    for per_text in generated_text:
        goldens.append(per_text["golden_response"])
        generations.append(per_text["generated_response"])
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
