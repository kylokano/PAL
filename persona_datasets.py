import torch
from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase, BertTokenizer
import json
import os
import pickle
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import numpy as np
import random
from utils import TemporarilySeededRandom
random.seed(42)


class BaiduPersonaChat(Dataset):
    """
    Baidu Personalized Dialogue Dataset Class
    
    Used for loading and processing Chinese personalized dialogue data, supporting multiple modes:
    - select: Select the most relevant personalized information
    - generate: Generate responses based on personalized information
    - generate_for_dpo: Generate data for DPO training
    - predict_select_then_generate: Predict selection then generate
    
    Args:
        dataset_version: Dataset version
        data_part: Data part (train, valid#1, valid#2, test)
        mode: Processing mode
        single_turn: Whether single turn dialogue
        with_prompt: Whether to include prompts
        only_persona_response: Whether to return only personalized responses
        small_data: Whether to use small dataset
        silent: Whether silent mode
        sft_mode: Whether SFT mode
        select_file_path: Selection file path
        data_dir: Data directory path
        **kwargs: Other parameters
    """
    
    def __init__(self, dataset_version: str, data_part: str, mode: str, single_turn: bool, 
                 with_prompt: bool, only_persona_response: bool, small_data: bool = False, 
                 silent: bool = False, sft_mode: bool = True, select_file_path: str = "", 
                 data_dir: str = None, **kwargs) -> None:
        super().__init__()
        
        if data_dir is None:
            data_dir = kwargs.get("chinese_data_dir", "./data")
        
        assert data_part in ["train", "valid#1", "valid#2", "test"]
        path = os.path.join(data_dir, f"{data_part}_ch.txt")
        print(f"Loading chinese {data_part} dataset, from {path}")
        
        with open(path, 'r', encoding='utf-8') as r:
            data = json.load(r)
        
        if kwargs.get("is_llama3") is True:
            prefix_token, end_token = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            prefix_token, end_token = "", ""
        
        if not sft_mode:
            print("Generate data for dpo")
            reject_data = json.load(open(kwargs["dpo_path"], 'r', encoding='utf8'))
            reject_dict = {}
            for item in reject_data:
                try:
                    reject_dict[item["prompt"]] = item["generated_response"]
                except:
                    print(item)
        
        if mode.startswith("predict_select_then_generate"):
            if mode == "predict_select_then_generate":
                if len(select_file_path) == 0:
                    select_data_path = kwargs.get("chinese_select_data_path", "./generated/ch_gpt2-ft-withprompt-generate_for_select_generate_select.json")
                else:
                    select_data_path = select_file_path
            else:
                raise KeyError("Check mode must be predict_select_then_generate")
            
            print("Load predicted persona via", select_data_path)
            predict_select = json.load(open(select_data_path, 'r', encoding='utf8'))
            predicted_select_dict = {}
            for item in predict_select:
                try:
                    predicted_select_dict[item["prompt"]] = item["generated_response"]
                except:
                    print(item)
            print(f"Select then Generate Mode, Load Select file at {select_data_path}")
        
        dpo_type_data = defaultdict(lambda: defaultdict(list))
        index = 0
        
        for i, (persona_list, history, response, persona_label) in tqdm(enumerate(data), desc="persona_chat", disable=silent):
            if single_turn:
                raise KeyError("Cannot be single turn")
            
            if only_persona_response and persona_label[0] == -1:
                personal_entry = ["无需个性化信息。"]
            elif only_persona_response:
                personal_entry = [persona_list[persona_label[0]]]
            else:
                personal_entry = persona_list
            
            if not with_prompt:
                raise KeyError("cannot set with_prompt False")
            
            personal_index, personal_description = 1, ""
            for per_persona in personal_entry:
                personal_description += f" {personal_index}.{per_persona}"
                personal_index += 1
            
            prefix_prompt = f"用户的个性化信息为：\"{personal_description}\".\n"
            dialogue_history = []
            for e in range(len(history)):
                if e % 2 == 0:
                    dialogue_history.append(f"用户1: {history[e]}")
                else:
                    dialogue_history.append(f"用户2: {history[e]}")
            dialogue_history = "\n".join(dialogue_history)
            which_persona = "用户1: " if len(history) == 0 else "用户2: "
            
            def get_prompt_label(mode):
                if mode == "select":
                    prompt = '''你是一个有用的、细致的对话代理。假设有一个用户在和你聊天。提供给你一组个性化信息描述和一个对话历史。
你的任务是首先查看对话历史，然后确定是否某个个性化信息描述与生成你的下一个回复相关。
如果没有个性化信息描述相关，请回答“没有个性化信息”。
如果恰好有一个个性化信息描述相关，请提供那个个性化信息描述。
个性化信息描述：<personas>。
对话历史：<dialogue context>。
'''
                    prompt = prompt.replace("<personas>", f"'''{persona_list}'''")
                    prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                    prompt = prefix_token + prompt + end_token
                    
                    try:
                        label = str(persona_label[0]+1) + "。" + persona_list[persona_label[0]] if persona_label[0] > -1 else "无需个性化信息。"
                    except:
                        label = "无需个性化信息。"
                        
                elif mode == "generate":
                    prompt = '''你是一个有用的、细致的对话代理。假设有一个用户在和你聊天。提供给你一组个性化信息描述和一个对话历史。你同时被指出了一个和你的回复最相关的个性化信息描述。如果这个描述是“无需个性化信息”，那么你就不需要使用个性化信息进行回复。
你的任务是依据这个最相关的个性化信息描述生成对话的回复。
个性化信息描述：<personas>。
对话历史：<dialogue context>。
最相关的个性化信息描述：<related_persona>。
'''
                    prompt = prompt.replace("<personas>", f"'''{persona_list}'''")
                    prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                    prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                    label = response
                    
                elif mode == "generate_for_dpo":
                    prompt = '''你是一个有用的、细致的对话代理。假设有一个用户在和你聊天。提供给你对话历史。
你的任务是生成对话的回复。
对话历史：<dialogue context>。
'''
                    prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                    prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                    label = response
                    
                elif mode == "predict_select_then_generate":
                    prompt = '''你是一个有用的、细致的对话代理。假设有一个用户在和你聊天。提供给你一组个性化信息描述和一个对话历史。你同时被指出了一个和你的回复最相关的个性化信息描述。如果这个描述是“无需个性化信息”，那么你就不需要使用个性化信息进行回复。
你的任务是依据这个最相关的个性化信息描述生成对话的回复。
个性化信息描述：<personas>。
对话历史：<dialogue context>。
最相关的个性化信息描述：<related_persona>。
'''
                    prompt = prompt.replace("<personas>", f"'''{persona_list}'''")
                    prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                    prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                    label = response

                return prompt, label
            
            prompt, label = get_prompt_label(mode)

            if not sft_mode:
                dpo_index_prompt, dpo_label = get_prompt_label(kwargs["dpo_data_mode"])
                dpo_responses, dpo_pairs = [label], []
                dpo_responses.append(reject_dict[dpo_index_prompt])
                dpo_pairs.append((0, 1))
                
            if not sft_mode:
                dpo_type_data[prompt]["pairs"] = dpo_pairs
                dpo_type_data[prompt]['responses'] = dpo_responses
            else:
                dpo_type_data[prompt]["pairs"] = None
                dpo_type_data[prompt]['responses'] = None

            dpo_type_data[prompt]['sft_target'] = label

            index += 1
            if small_data and index > 50:
                break
        
        self.data = dpo_type_data
        if not silent:
            print(f"Total {data_part} data number:{len(self.data)}") if not small_data else print(
                f"Small {data_part} data number:{len(self.data)}")


class PersonaChat(Dataset):
    """
    English Personalized Dialogue Dataset Class
    
    Used for loading and processing English personalized dialogue data, supporting multiple modes:
    - select: Select the most relevant personalized information
    - generate: Generate responses based on personalized information
    - generate_for_dpo: Generate data for DPO training
    - predict_select_then_generate: Predict selection then generate
    
    Args:
        dataset_version: Dataset version
        data_part: Data part (train, valid#1, valid#2, test)
        mode: Processing mode
        single_turn: Whether single turn dialogue
        with_prompt: Whether to include prompts
        only_persona_response: Whether to return only personalized responses
        small_data: Whether to use small dataset
        silent: Whether silent mode
        sft_mode: Whether SFT mode
        select_file_path: Selection file path
        data_dir: Data directory path
        **kwargs: Other parameters
    """
    
    def __init__(self, dataset_version: str, data_part: str, mode: str, single_turn: bool, 
                 with_prompt: bool, only_persona_response: bool, small_data: bool = False, 
                 silent: bool = False, sft_mode: bool = True, select_file_path: str = "",
                 data_dir: str = "./data", **kwargs) -> None:
        super().__init__()
        
        assert data_part in ["train", "valid#1", "valid#2", "test"], data_part
        assert dataset_version in ["origin", "revised"], dataset_version
        
        if kwargs.get("is_llama3") is True:
            prefix_token, end_token = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            prefix_token, end_token = "", ""
        
        if data_part in ["train", "valid"]:
            if sft_mode:
                path = os.path.join(data_dir, f"{data_part}_{dataset_version}.txt")
            else:
                path = os.path.join(data_dir, f"{data_part}_{dataset_version}_dpo_train.txt")
        else:
            path = os.path.join(data_dir, f"{data_part}.txt")
        
        print(f"Loading {data_part} dataset, from {path}")
        
        with open(path, 'r', encoding='utf-8') as r:
            data = json.load(r)

        if not sft_mode:
            print("Generate data for dpo")
            reject_data = json.load(open(kwargs["dpo_path"], 'r', encoding='utf8'))
            reject_dict = {}
            for item in reject_data:
                try:
                    reject_dict[item["prompt"]] = item["generated_response"]
                except:
                    print(item)

        if mode.startswith("predict_select_then_generate"):
            if len(select_file_path) == 0:
                if mode == "predict_select_then_generate":
                    select_data_path = kwargs.get("english_select_data_path", "./generated/gpt2-ft-withprompt-mixture-select_generate_select.json")
                else:
                    tmp_m = mode.split("predict_select_then_generate")[1]
                    print(f"Using {tmp_m}")
                    if tmp_m == "dialogpt":
                        select_data_path = kwargs.get("dialogpt_select_data_path", "./generated/dialogpt-ft-withprompt-reverse-generate_for_dpo_generate_select.json")
                    elif tmp_m == "gpt2":
                        select_data_path = kwargs.get("gpt2_select_data_path", "./generated/gpt2-ft-withprompt-reverse-generate_for_dpo_generate_select.json")
                    else:
                        raise KeyError("Please check data_mode")
                    mode = "predict_select_then_generate"
            else:
                select_data_path = select_file_path
            print(f"Select then Generate Mode, Load Select file at {select_data_path}")
            predict_select = json.load(open(select_data_path, 'r', encoding='utf8'))
            predicted_select_dict = {}
            for item in predict_select:
                try:
                    predicted_select_dict[item["prompt"]] = item["generated_response"]
                except:
                    print(item)

        dpo_type_data = defaultdict(lambda: defaultdict(list))
        index = 0
        
        for i, (persona_list, history, response, persona_label) in tqdm(enumerate(data), desc="persona_chat", disable=silent):
            if single_turn:
                history = [history[-1]]
            if only_persona_response and persona_label[0] == -1:
                personal_entry = ["No persona data needed . "]
            elif only_persona_response:
                personal_entry = [persona_list[persona_label[0]]]
            else:
                personal_entry = persona_list

            if with_prompt:
                personal_index, personal_description = 1, ""
                for per_persona in personal_entry:
                    personal_description += f" {personal_index}.{per_persona}"
                    personal_index += 1
                
                dialogue_history = []
                for e in range(len(history)):
                    if e % 2 == 0:
                        dialogue_history.append(f"person1: {history[e]}")
                    else:
                        dialogue_history.append(f"person2: {history[e]}")
                dialogue_history = "\n".join(dialogue_history)
                which_persona = "person1: " if len(history) == 0 else "person2: "

                def get_prompt_label(mode):
                    if mode == "select":
                        prompt = '''You are a helpful and thorough conversational agent. Assume there is a user chatting with you. You are given a set of personal descriptions and a dialogue history.
Your task is first to review the dialogue history and then determine if any one personal description from the personal descriptions set is relevant to formulating your next response.
If no personal descriptions are relevant, respond with "No personal description."
If exactly one personal description is relevant, provide only that personal description.
The personal descriptions set: <personas>.
The dialogue history: <dialogue context>.
'''
                        prompt = prompt.replace("<personas>", f"'''{personal_description}'''")
                        prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                        prompt = prefix_token + prompt + end_token
                        try:
                            label = str(persona_label[0]+1) + "." + persona_list[persona_label[0]] if persona_label[0] > -1 else "No personal description."
                        except:
                            label = "No personal description."
                            
                    elif mode == "generate":
                        prompt = '''You are a helpful and thorough conversational agent. Assume there is a user chatting with you. You are given a set of personal descriptions and a dialogue history. You are also given one highlighted personal description from that list, which is most relevant to your next response. If the highlighted personal description is “No personal description,” it means that none of the personal descriptions is appropriate for the next response. 
Your task is to provide the next turn of the dialogue using the highlighted personal description when available.
The personal descriptions set: <personas>.
The dialogue history: <dialogue context>.
The highlighted personal description: <related_persona>.
'''
                        prompt = prompt.replace("<personas>", f"'''{personal_description}'''")
                        prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                        try:
                            p_label = str(persona_label[0]+1) + "." + persona_list[persona_label[0]] if persona_label[0] > -1 else "No personal description."
                        except:
                            p_label = "No personal description."
                        prompt = prompt.replace("<related_persona>", f"'''{p_label}'''")
                        prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                        label = response
                        
                    elif mode == "generate_for_dpo":
                        prompt = '''You are a helpful and thorough conversational agent. You have a dialogue history. 
You should provide the next turn of the dialogue based on the dialogue history. 
The dialogue history is: <dialogue context>.
'''
                        prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                        prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                        label = response
                        
                    elif mode == "predict_select_then_generate":
                        prompt = '''You are a helpful and thorough conversational agent. Assume there is a user chatting with you. You are given a set of personal descriptions and a dialogue history. You are also given one highlighted personal description from that list, which is most relevant to your next response. If the highlighted personal description is "No personal description," it means that none of the personal descriptions is appropriate for the next response. 
Your task is to provide the next turn of the dialogue using the highlighted personal description when available.
The personal descriptions set: <personas>.
The dialogue history: <dialogue context>.
The highlighted personal description: <related_persona>.
'''
                        prompt = prompt.replace("<personas>", f"'''{personal_description}'''")
                        prompt = prompt.replace("<dialogue context>", f"'''{dialogue_history}'''")
                        try:
                            p_label = str(persona_label[0]+1) + "." + persona_list[persona_label[0]] if persona_label[0] > -1 else "No personal description."
                        except:
                            p_label = "No personal description."
                        prompt = prompt.replace("<related_persona>", f"'''{p_label}'''")
                        prompt = prefix_token + prompt + end_token + f"\n{which_persona}"
                        label = response

                    return prompt, label

                prompt, label = get_prompt_label(mode)

                if not sft_mode:
                    dpo_index_prompt, dpo_label = get_prompt_label(kwargs["dpo_data_mode"])
                    dpo_responses, dpo_pairs = [label], []
                    dpo_responses.append(reject_dict[dpo_index_prompt])
                    dpo_pairs.append((0, 1))
            else:
                personal_description = " ".join(personal_entry)
                dialogue_history = " ".join(history)
                if mode == "select":
                    prompt = f"{personal_description}. {dialogue_history}"
                    label = persona_list[persona_label[0]] if persona_label[0] > -1 else "No persona data needed"
                elif mode == "generate":
                    prompt = f"{personal_description}. {dialogue_history} Response"
                    label = response
                else:
                    raise KeyError(f"Unknown mode {mode} without prompt. Please check the data_mode or with_prompt parameter")

            if not sft_mode:
                dpo_type_data[prompt]["pairs"] = dpo_pairs
                dpo_type_data[prompt]['responses'] = dpo_responses
            else:
                dpo_type_data[prompt]["pairs"] = None
                dpo_type_data[prompt]['responses'] = None

            dpo_type_data[prompt]['sft_target'] = label

            index += 1
            if small_data and index > 50:
                break

        self.data = dpo_type_data
        if not silent:
            print(f"Total {data_part} data number:{len(self.data)}") if not small_data else print(
                f"Small {data_part} data number:{len(self.data)}")


def get_dataset(name: str, data_part: str, mode: str, single_turn: bool, with_prompt: bool, 
                only_persona_response: bool, small_data: bool, silent: bool, sft_mode: bool, 
                is_chinese: bool, select_file_path: str, data_dir: str = None, **kwargs):
    """
    Get dataset
    
    Args:
        name: Dataset name
        data_part: Data part
        mode: Processing mode
        single_turn: Whether single turn dialogue
        with_prompt: Whether to include prompts
        only_persona_response: Whether to return only personalized responses
        small_data: Whether to use small dataset
        silent: Whether silent mode
        sft_mode: Whether SFT mode
        is_chinese: Whether Chinese dataset
        select_file_path: Selection file path
        data_dir: Data directory path
        **kwargs: Other parameters
        
    Returns:
        Processed dataset
    """
    if is_chinese:
        Datas = BaiduPersonaChat
        if data_dir is None:
            data_dir = kwargs.get("chinese_data_dir", "./data")
    else:
        Datas = PersonaChat
        if data_dir is None:
            data_dir = kwargs.get("english_data_dir", "./data")
            
    if name.startswith("persona_chat"):
        print(f"get_dataset select file path {select_file_path}")
        dataset_version = name.split("_")[-1]
        if mode == "mixture":
            select_data = Datas(dataset_version, data_part, "select", single_turn,
                                      with_prompt, only_persona_response, small_data, silent, sft_mode, select_file_path, data_dir=data_dir, **kwargs).data
            generate_data = Datas(dataset_version, data_part, "generate", single_turn,
                                        with_prompt, only_persona_response, small_data, silent, sft_mode, select_file_path, data_dir=data_dir, **kwargs).data
            data = {**select_data, **generate_data}
        elif mode.startswith("mixture_"):
            continue_ratio = float(mode.split("mixture_")[-1])
            print("Only keep the ratio of ", continue_ratio, " select data")
            select_data = Datas(dataset_version, data_part, "select", single_turn,
                                      with_prompt, only_persona_response, small_data, silent, sft_mode, select_file_path, data_dir=data_dir, **kwargs).data
            no_persona_select_keys = []
            for prompt, item in select_data.items():
                if item["sft_target"] == "No persona data needed":
                    no_persona_select_keys.append(prompt)
            for per_prompt in no_persona_select_keys:
                p = random.random()
                if p > continue_ratio:
                    del select_data[per_prompt]
            generate_data = Datas(dataset_version, data_part, "generate", single_turn,
                                        with_prompt, only_persona_response, small_data, silent, sft_mode, select_file_path, data_dir=data_dir, **kwargs).data
            data = {**select_data, **generate_data}
        else:
            data = Datas(dataset_version, data_part, mode, single_turn,
                               with_prompt, only_persona_response, small_data, silent, sft_mode, select_file_path, data_dir=data_dir, **kwargs).data
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """
    Return collate function for given tokenizer
    
    The collate function receives a list of examples (dictionaries where values are int lists [tokens] or strings [raw text]),
    and returns a batch of examples, PyTorch tensors padded to maximum length. Strings are passed directly.
    
    Args:
        tokenizer: Tokenizer
        
    Returns:
        Collate function
    """
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    if tokenizer.pad_token_id is not None:
                        padding_value = tokenizer.pad_token_id
                    else:
                        padding_value = 0
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, 
                          tokenizer, max_length: int, max_prompt_length: int, dtype: str = "bfloat16") -> Dict:
    """
    Tokenize a single batch element
    
    At this stage, we don't convert to PyTorch tensors yet; we only handle truncation,
    in case prompt + chosen or prompt + rejected response is too long.
    First truncate prompt; if still too long, we truncate chosen/rejected.
    
    We also create labels for chosen/rejected responses, with length equal to the sum of prompt and chosen/rejected response lengths,
    with prompt tokens set to -100.
    
    Args:
        prompt: Prompt text
        chosen: Chosen response
        rejected: Rejected response
        truncation_mode: Truncation mode
        tokenizer: Tokenizer
        max_length: Maximum length
        max_prompt_length: Maximum prompt length
        dtype: Data type
        
    Returns:
        Tokenized batch element
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"
    
    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(
        len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length]
                             for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:]
                             for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length]
                         for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length]
                           for k, v in rejected_tokens.items()}

    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(
        prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(
        prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       select_file_path: str = "",
                       datasets_kwags: Optional[dict] = None) -> Iterator[Dict]:
    """
    Get data batch iterator
    
    Stops when either n_epochs or n_examples is reached first.
    
    Args:
        names: List of dataset names to use
        tokenizer: Tokenizer to use
        split: Split to use
        batch_size: Batch size
        shuffle: Whether to shuffle data after each epoch
        max_length: Maximum length of prompt + response combination
        max_prompt_length: Maximum length of prompt
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected)
        n_epochs: Number of epochs to run. Must specify this parameter or n_examples
        n_examples: Number of examples to run. Must specify this parameter or n_epochs
        seed: Random seed
        silent: Whether to silence progress bar
        select_file_path: Selection file path
        datasets_kwags: Dataset keyword arguments
        
    Returns:
        Data batch iterator
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        print(f"get_batch_iterator select file path{select_file_path}")
        for name in names:
            truncation_mode = 'keep_end'
            for prompt, data in get_dataset(name, data_part=split, silent=silent, sft_mode=sft_mode, select_file_path=select_file_path, **datasets_kwags).items():
                flat_data.append(
                    (prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            random_seed = next(permutation_seeds)
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                assert sft_target is not None
                batch_element = tokenize_batch_element(
                    prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {
                    k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                assert pairs is not None and responses is not None
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(
                        prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1