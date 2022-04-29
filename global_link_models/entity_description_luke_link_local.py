import random
from typing import Optional, List, Tuple, Dict, NamedTuple, Union
from typing_extensions import Literal, final
from pathlib import Path
from logging import Logger
from itertools import chain
import re

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import einops as eop

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer, PretrainedConfig
from torch.nn import LayerNorm as BertLayerNorm
from transformers.models.bert import BertModel
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device


from emb_distill_models.entity_discription_luke import EntityDescriptionLukeBase
from global_link_models.model_type_hints import BaseModelReturn
from global_link_dataset import ModelExample, WordBertInput, EntityBertInput, EntityMLMLabel, WordMLMLabel
from utils.freeze_model import freeze, un_freeze


class EntityDescriptionLukeLinkLocal(EntityDescriptionLukeBase):
    allow_manually_init = False

    def __init__(
            self,
            # 自己需要的参数
            out_topk_candidates: int,
            entity_encoding_path: Union[Path, str],
            candidate_encoding_device: Optional[str],
            candidate_from: Literal["full", "list"],
            normalize_entity_encoding: bool,
            # 不是参数
            py_logger: Logger,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            # 父类需要的参数
            bert_pretrained_dir: Union[Path, str],
            luke_pretrained_dir: Union[Path, str],
            share_backbone: bool,
            entity_encoder_use_luke: bool,
            special_entity_emb_size: int,
            entity_pooling: Literal["average", "cls", "luke_span"],
            do_transform: bool,

            *args,
            **kwargs
    ):
        # warning: 子类中不要新增含权重的layer

        super().__init__(
            bert_pretrained_dir=bert_pretrained_dir,
            luke_pretrained_dir=luke_pretrained_dir,
            share_backbone=share_backbone,
            entity_encoder_use_luke=entity_encoder_use_luke,
            special_entity_emb_size=special_entity_emb_size,
            entity_pooling=entity_pooling,
            context_init_checkpoint=None,
            entity_init_checkpoint=None,
            py_logger=py_logger,
            do_transform=do_transform,
            *args,
            **kwargs
        )

        self.tokenizer = tokenizer
        self.out_topk_candidates = out_topk_candidates
        self.candidate_from = candidate_from

        if self.py_logger:
            self.py_logger.info(f"loading precomputed candidate encoding: {entity_encoding_path}")
        precomputed_candidate_encoding: torch.Tensor = torch.load(
            entity_encoding_path, map_location="cpu"
        )  # 非参数的tensor不会被pl.Trainer自动迁移到模型所在cuda上
        self.candidate_encoding_device = torch.device(candidate_encoding_device) if candidate_encoding_device is not None else None
        self.precomputed_candidate_encoding = torch.cat([
            torch.zeros((4, precomputed_candidate_encoding.shape[1]), dtype=precomputed_candidate_encoding.dtype),
            precomputed_candidate_encoding,
        ], dim=0).to(candidate_encoding_device)
        if normalize_entity_encoding:
            self.precomputed_candidate_encoding = F.normalize(self.precomputed_candidate_encoding, p=2, dim=1)

        self.train_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.eval_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")
        pass

    def forward(self, batch: ModelExample):

        if self.candidate_encoding_device is None:
            if self.precomputed_candidate_encoding.device != self.device:
                self.precomputed_candidate_encoding = self.precomputed_candidate_encoding.to(self.device)
        else:
            if self.precomputed_candidate_encoding.device != self.candidate_encoding_device:
                self.precomputed_candidate_encoding = self.precomputed_candidate_encoding.to(self.candidate_encoding_device)


        ######################################
        #   encode context and mention
        ######################################

        # # # encode

        ctxt_word_seq_output, ctxt_entity_seq_output = self.encode_context_word_entity_sequence(
            word_seq=batch.word_seq, entity_seq=batch.entity_seq)
        # ctxt_word_seq_output: [batch_size, word_seq_len, hidden_size]
        # ctxt_entity_seq_output: [batch_size, max_entity_num, hidden_size]

        # # # select masked entity

        # entity_mask: [batch_size, max_entity_num]
        target_ctxt_entity_seq_output = torch.masked_select(
            ctxt_entity_seq_output, batch.masked_entity_label.mlm_mask.bool().unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_ctxt_entity_seq_output: [batch_mask_entity_num * hidden_size]

        context_hidden_size = ctxt_entity_seq_output.shape[-1]
        target_ctxt_entity_seq_output = target_ctxt_entity_seq_output.view(-1, context_hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ###########################################
        #   get entity mask
        ###########################################

        # masked_entity_label.mlm_label: [batch_size, max_entity_num]
        entity_mask = batch.masked_entity_label.mlm_label != -100
        # entity_mask: [batch_size, max_entity_num]
        assert (entity_mask == batch.masked_entity_label.mlm_mask).all()

        ###########################################
        #   score
        ###########################################

        if self.candidate_from == 'full':
            model_return = self.score_from_full_matrix(
                masked_entity_label=batch.masked_entity_label,
                entity_sequence_output=ctxt_entity_seq_output,
            )
        elif self.candidate_from == 'list':
            model_return = self.score_from_candidate_list(
                masked_entity_label=batch.masked_entity_label,
                entity_sequence_output=ctxt_entity_seq_output,
            )
        else:
            raise ValueError(self.candidate_from)

        return model_return

    def entity_predictions(
            self,
            hidden_states: torch.Tensor,
            list_cand_emb_index: Optional[List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        hidden_states = hidden_states.to(self.precomputed_candidate_encoding.device)
        if list_cand_emb_index is not None:
            scores = []
            for cand_emb_index, hid_st in zip(list_cand_emb_index, hidden_states.unbind(dim=0)):
                cand_emb = self.precomputed_candidate_encoding[cand_emb_index]
                s = torch.einsum("h, c h -> c", hid_st, cand_emb)  # [cand_num]
                scores.append(s)
        else:
            scores = torch.einsum("e h, v h -> e v", hidden_states, self.precomputed_candidate_encoding)
            # [entity_num, vocab_size]
        return scores

    def score_from_candidate_list(
            self,
            masked_entity_label: EntityMLMLabel,
            entity_sequence_output: torch.Tensor,
    ):

        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility
        device = next(iter(self.parameters())).device

        entity_mask = masked_entity_label.mlm_mask.bool()

        ###########################################
        #   select masked entity span
        ###########################################

        target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_entity_sequence_output: [batch_mask_entity_num * hidden_size]

        target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ###########################################
        #  convert candidate to emb_index_list
        ###########################################

        # batch.masked_entity_label.candidate_vocab_index:
        #   Tuple[Tuple[slice]] or Tuple[Tuple[List[int]]], len: batch_size * mention_num * cand_num
        list_candidate_emb_index = list(chain.from_iterable(masked_entity_label.candidate_vocab_index))
        # List[slice] or List[List[int]], len: batch_mask_entity_num * cand_num

        all_emb_index = torch.tensor(range(self.entity_vocab_size()), device=device)
        list_candidate_emb_index = [
            all_emb_index[cand_emb_index] if isinstance(cand_emb_index, slice) else torch.tensor(cand_emb_index, device=device)
            for cand_emb_index in list_candidate_emb_index
        ]
        # List[tensor], len: batch_mask_entity_num * [cand_num]

        ###########################################
        #   predict for each mention
        ###########################################

        list_cand_scores = self.entity_predictions(
            hidden_states=target_entity_sequence_output, list_cand_emb_index=list_candidate_emb_index)
        # List[Tensor], batch_size * [cand_num_i]

        # padding for loss function
        pad_cand_scores = pad_sequence(
            sequences=list_cand_scores, batch_first=True, padding_value=-10_000.)
        # [batch_size, max_cand_num]

        ###########################################
        #   compute loss
        ###########################################

        # example:
        #   candidate emb_index: [
        #       [10, 20, 30],
        #       [21, 42],
        #       [52, 12, 42, 61]
        #   ]
        #   label: [20, 21, 61]  ->  [1, 0, 3]

        #   emb_index to candidate list label index
        target_entity_labels = torch.masked_select(masked_entity_label.mlm_label, entity_mask)
        # [batch_mask_entity_num]

        list_cand_num = [len(cand_emb_index) for cand_emb_index in list_candidate_emb_index]  # len: batch_size
        label_cand_list_index = torch.tensor([
            cand_emb_index.tolist().index(label_emb_index)
            for cand_emb_index, label_emb_index in zip(list_candidate_emb_index, target_entity_labels)
        ], device=device)  # [batch_size]

        if self.training:
            loss = self.train_loss(input=pad_cand_scores, target=label_cand_list_index)
        else:
            loss = self.eval_loss(input=pad_cand_scores, target=label_cand_list_index)

        ###########################################
        #   get topk
        ###########################################

        list_topk_cand_score_index = [
            torch.argsort(cand_scores, descending=True)
            for cand_scores in list_cand_scores
        ]
        list_topk_scores = [
            cand_scores[topk_index][:self.out_topk_candidates]
            for topk_index, cand_scores in zip(list_topk_cand_score_index, list_cand_scores)
        ]  # batch_mask_entity_num * [min(out_top_k, cand_num)]
        list_topk_indices = [
            cand_emb_index[topk_index][:self.out_topk_candidates]
            for topk_index, cand_emb_index in zip(list_topk_cand_score_index, list_candidate_emb_index)
        ]  # batch_mask_entity_num * [min(out_top_k, cand_num)]

        # 注：实验发现，不在词表里的词，有很大比例会被链接到[UNK]，也就是1

        ###########################################
        #   prepare return
        ###########################################

        model_return = BaseModelReturn(
            query_id=list(chain.from_iterable(masked_entity_label.query_id)),
            has_gold=list(chain.from_iterable(masked_entity_label.has_gold)),
            freq_bin=list(chain.from_iterable(masked_entity_label.luke_freq_bin)),
            gold_wikidata_id=list(chain.from_iterable(masked_entity_label.gold_wikidata_id)),
            gold_wikipedia_pageid=list(chain.from_iterable(masked_entity_label.gold_wikipedia_pageid)),
            label=target_entity_labels,
            predict_score=list_cand_scores,
            out_topk_predict_score=list_topk_scores,
            out_topk_indices=list_topk_indices,
            loss=loss,
        )
        return model_return

    def score_from_full_matrix(
            self,
            masked_entity_label: EntityMLMLabel,
            entity_sequence_output: torch.Tensor,
    ):
        entity_mask = masked_entity_label.mlm_mask.bool()

        ###########################################
        #   select masked entity span
        ###########################################

        target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_entity_sequence_output: [batch_mask_entity_num * hidden_size]

        context_hidden_size = entity_sequence_output.shape[-1]
        target_entity_sequence_output = target_entity_sequence_output.view(-1, context_hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ###########################################
        #   pred score for each mention
        ###########################################

        # predict on full entity emb matrix
        all_pred_entity_scores = self.entity_predictions(
            hidden_states=target_entity_sequence_output,
            list_cand_emb_index=None,
        )  # [batch_mask_entity_num, entity_vocab_size]
        # all_pred_entity_scores = all_pred_entity_scores.view(-1, self.config.entity_vocab_size)  # [batch_mask_entity_num, entity_vocab_size]

        # select candidate score only
        entity_scores = torch.ones_like(all_pred_entity_scores) * -10_000.  # [batch_mask_entity_num, entity_vocab_size]
        list_candidate_emb_index = list(chain.from_iterable(masked_entity_label.candidate_vocab_index))
        assert len(list_candidate_emb_index) == entity_scores.shape[0]
        for i, c in enumerate(list_candidate_emb_index):
            c: Union[slice, List]
            entity_scores[i][c] = all_pred_entity_scores[i][c]

        ###########################################
        #   compute loss
        ###########################################

        target_entity_labels = torch.masked_select(masked_entity_label.mlm_label, entity_mask)
        # [batch_mask_entity_num]

        if self.training:
            loss = self.train_loss(input=entity_scores, target=target_entity_labels)
        else:
            loss = self.eval_loss(input=entity_scores, target=target_entity_labels)

        ###########################################
        #   get topk
        ###########################################

        topk_values, topk_indices = entity_scores.topk(k=self.out_topk_candidates, dim=1)
        # topk_values, topk_indices: [batch_mask_entity_num, out_top_k]

        ###########################################
        #   prepare return
        ###########################################

        model_return = BaseModelReturn(
            query_id=list(chain.from_iterable(masked_entity_label.query_id)),
            has_gold=list(chain.from_iterable(masked_entity_label.has_gold)),
            freq_bin=list(chain.from_iterable(masked_entity_label.luke_freq_bin)),
            gold_wikidata_id=list(chain.from_iterable(masked_entity_label.gold_wikidata_id)),
            gold_wikipedia_pageid=list(chain.from_iterable(masked_entity_label.gold_wikipedia_pageid)),
            label=target_entity_labels,
            predict_score=entity_scores,
            out_topk_predict_score=topk_values,
            out_topk_indices=topk_indices,
            loss=loss,
        )

        return model_return

    def entity_vocab_size(self):
        return self.precomputed_candidate_encoding.shape[0]


