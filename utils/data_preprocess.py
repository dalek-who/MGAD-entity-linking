from typing import Dict, Tuple, List, Union, Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer, \
    PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer

import torch
import numpy as np

def get_mention_context(context_left: List[int], context_right: List[int], mention: List[int], max_context_len: int):
    """
    截取mention的上下文，尽量把mention放在中间位置
    mention、context都是已经分词后的id list
    :param context_left:
    :param context_right:
    :param mention:
    :param max_context_len:
    :return:
    """
    assert len(mention) <= max_context_len, \
        (len(mention), mention)
    left_quota = (max_context_len - len(mention)) // 2 - 1  # mention左边的长度配额
    right_quota = max_context_len - len(mention) - left_quota - 2  # mention右边长度配额。2是给cls和sep的
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:  # 依据左右实际长度，修正实际需要的配额
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    mention_start = len(context_left[-left_quota:])
    mention_end = len(context_left[-left_quota:] + mention) - 1
    context = context_left[-left_quota:] + mention + context_right[:right_quota]
    assert len(context) <= max_context_len, \
        (len(context), max_context_len)
    return context, mention_start, mention_end


def get_entity_text(title: List[int], description: List[int], max_entity_len: int, title_token_id: int):
    """
    单个获取entity的文本表示
    :param title:
    :param description:
    :param max_entity_len:
    :param title_token_id:
    :return:
    """
    truncate_token_num = max(len(title) + len(description) + 3 - max_entity_len, 0)  # 3：cls，sep，title
    entity_text = description[:-truncate_token_num] + [title_token_id] + title
    return entity_text

# def to_bert_input(cls_id: int, sep_id: int, text1: List[int], max_seq_len: int, device=None, additional_text_list: List[List[int]]=None):
#     bert_input = dict()
#     t1 = [cls_id] + text1 + [sep_id]
#     bert_input["input_ids"] = t1
#     bert_input["token_type_ids"] = [0] * len(t1)
#     bert_input["attention_mask"] = [1] * len(t1)
#     # add additional text(maybe more than two)
#     if additional_text_list is not None:
#         for sid, text_i in enumerate(additional_text_list, start=1):
#             assert isinstance(text_i, list)
#             new_text = text_i + [sep_id]
#             bert_input["input_ids"] += new_text
#             bert_input["token_type_ids"] += [sid] * len(new_text)
#             bert_input["attention_mask"] += [1] * len(new_text)
#     assert len(bert_input["input_ids"]) == len(bert_input["token_type_ids"]) == len(bert_input["attention_mask"])
#     # truncate
#     if len(bert_input["input_ids"]) > max_seq_len:
#         bert_input["input_ids"] = bert_input["input_ids"][:max_seq_len-1] + [sep_id]
#         bert_input["token_type_ids"] = bert_input["token_type_ids"][:max_seq_len]
#         bert_input["attention_mask"] = bert_input["attention_mask"][:max_seq_len]
#     assert len(bert_input["input_ids"]) == len(bert_input["token_type_ids"]) == len(bert_input["attention_mask"]) <= max_seq_len
#     if device:
#         bert_input = {k: torch.tensor(v).to(device) for k, v in bert_input.items()}
#     return bert_input


def to_transformers_input(
        tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast,
        token_ids_0: List[int], max_seq_len: int, device=None,
        token_ids_1: List[int] = None,
        truncation_strategy="longest_first"
):
    """
    convert two token_id_list (without cls,sep) to transformer input with input_ids, token_type_ids, attention_mask
    :param tokenizer:
    :param token_ids_0:
    :param max_seq_len:
    :param device:
    :param token_ids_1:
    :param truncation_strategy:
    :return:
    """
    """
    truncation_strategy: 
        longest_first
        only_first
        only_second
        do_not_truncate
    """

    bert_input = dict()

    input_ids = tokenizer.build_inputs_with_special_tokens(
        token_ids_0=token_ids_0, token_ids_1=token_ids_1)
    if len(input_ids) > max_seq_len:  # truncate
        token_ids_0, token_ids_1, unused_ids = tokenizer.truncate_sequences(
            token_ids_0,
            token_ids_1,
            num_tokens_to_remove=len(input_ids)-max_seq_len,
            truncation_strategy=truncation_strategy,
        )
        input_ids = tokenizer.build_inputs_with_special_tokens(
            token_ids_0=token_ids_0, token_ids_1=token_ids_1)

    bert_input["input_ids"] = input_ids
    bert_input["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(
        token_ids_0=token_ids_0, token_ids_1=token_ids_1)
    assert len(bert_input["input_ids"]) == len(bert_input["token_type_ids"])
    bert_input["attention_mask"] = [1] * len(input_ids)
    if device:
        bert_input = {k: torch.tensor(v).to(device) for k, v in bert_input.items()}
    return bert_input


def pad_sequences(sequences: List[List], pad_token, max_seq_len: int):
    max_len = max([len(seq) for seq in sequences])
    padded_seq_len = min(max_len, max_seq_len)
    new_sequences = [
        seq + [pad_token] * (padded_seq_len-len(seq))
        for seq in sequences
    ]
    assert all([len(seq)==padded_seq_len for seq in new_sequences])
    return new_sequences


def tokenize_context(  # 这个比较复杂的tokenize过程，而不是直接用tokenizer.tokenize，是为了在需要truncate时，mention能放在比较中间的位置，左边、右边的context长度差不多
    mention: str,
    context_left: str,
    context_right: str,
    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast,
    max_context_length: int,
    ent_start_token: str,
    ent_end_token: str,
):
    assert mention
    mention_tokens = tokenizer.tokenize(mention)
    mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
    # if sample[mention_key] and len(sample[mention_key]) > 0:
    #     mention_tokens = tokenizer.tokenize(mention)
    #     mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_context_length - len(mention_tokens)) // 2 - 1  # mention左边的长度配额
    right_quota = max_context_length - len(mention_tokens) - left_quota - 2  # mention右边长度配额。2是给cls和sep的
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:  # 依据左右实际长度，修正实际需要的配额
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    context_str = " ".join(context_tokens)
    return tokenizer(context_str)

    # context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    # input_ids = tokenizer.convert_tokens_to_ids(context_tokens)

    # padding = [0] * (max_context_length - len(input_ids))
    # input_ids += padding
    # assert len(input_ids) == max_context_length
    #
    # return {
    #     "tokens": context_tokens,
    #     "ids": input_ids,
    # }
    # return input_ids

def tokenize_entity(
        title: str, description: str, ent_title_tag: str,
        tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast):
    assert title or description
    if description is not None:
        full_text = f"""{description} {ent_title_tag} {title}"""
    else:
        full_text = title
    return tokenizer(full_text)
