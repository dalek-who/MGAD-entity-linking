# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict, Tuple, NamedTuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig, BertModel

from transformers import AutoModel, AutoTokenizer, AutoConfig, \
    PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device

from link_dataset import ModelExample, BertInput
from link_models.model_type_hints import BaseModelReturn

@dataclass
class OnAnotherDevice:
    """
    保存在cpu上，不随模型其他部分迁移到gpu
    """
    tensor_on_device: torch.Tensor


class BertEncoder(nn.Module):
    def __init__(
            self,
            bert_model,
            output_dim,
            layer_pulled=-1,
            add_linear=None
    ):
        super(BertEncoder, self).__init__()
        print(f"bert_model_class: {bert_model.__class__}")
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        try:  # transformers
            bert_result: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
                token_ids, segment_ids, attention_mask
            )
            output_bert, output_pooler = bert_result.last_hidden_state, bert_result.pooler_output
        except AttributeError:  # pytorch-transformers
            output_bert, output_pooler = self.bert_model(
                token_ids, segment_ids, attention_mask
            )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result


class BiEncoderModule(torch.nn.Module):
    def __init__(
            self,
            pretrained_dir: Union[str, Path],
            out_dim: int,
            pull_from_layer,
            add_linear: bool,
            have_ctxt_encoder: bool,
            have_cand_encoder: bool,
    ):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(pretrained_dir)  # todo: AutoModel
        cand_bert = BertModel.from_pretrained(pretrained_dir)

        self.context_encoder = BertEncoder(
            bert_model=ctxt_bert,
            output_dim=out_dim,
            layer_pulled=pull_from_layer,
            add_linear=add_linear,
        ) if have_ctxt_encoder else None

        self.cand_encoder = BertEncoder(
            bert_model=cand_bert,
            output_dim=out_dim,
            layer_pulled=pull_from_layer,
            add_linear=add_linear,
        ) if have_cand_encoder else None

        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BlinkBiEncoderBase(pl.LightningModule):
    def __init__(
            self,
            pretrained_dir: Union[Path, str],
            out_dim: int,
            pull_from_layer,
            add_linear: bool,
            have_ctxt_encoder: bool,
            have_cand_encoder: bool,
            *args,
            **kwargs,
    ):
        super().__init__()

        # init model
        self.model = BiEncoderModule(
            pretrained_dir=pretrained_dir,
            out_dim=out_dim,
            pull_from_layer=pull_from_layer,
            add_linear=add_linear,
            have_ctxt_encoder=have_ctxt_encoder,
            have_cand_encoder=have_cand_encoder,
        )
        pass

    def entity_vocab_size(self):
        raise NotImplementedError()

    def encode_context(self, context: BertInput):
        # Encode contexts first
        embedding_ctxt, _ = self.model(
            # context
            token_idx_ctxt=context.input_ids,
            segment_idx_ctxt=context.token_type_ids,
            mask_ctxt=context.attention_mask,
            # candidate
            token_idx_cands=None,
            segment_idx_cands=None,
            mask_cands=None,
        )
        # embedding_ctxt: [batch_size, emb_dim]
        return embedding_ctxt

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class BlinkBiEncoderRanker(BlinkBiEncoderBase):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast,
            out_topk_candidates: int,
            entity_encoding_path: Union[Path, str],
            candidate_encoding_device: Optional[str],
            py_logger: logging.Logger=None,
            *args,
            **kwargs,
    ):
        super().__init__(
            have_ctxt_encoder=True, have_cand_encoder=False,
            *args, **kwargs
        )

        self.out_topk_candidates = out_topk_candidates
        self.tokenizer = tokenizer
        self.py_logger = py_logger

        # init model
        if self.py_logger:
            self.py_logger.info("loading precomputed candidate encoding")

        self.candidate_encoding_device = torch.device(candidate_encoding_device) if candidate_encoding_device is not None else None
        self.precomputed_candidate_encoding = torch.load(
            entity_encoding_path, map_location="cpu"
        )  # 非参数的tensor不会被pl.Trainer自动迁移到模型所在cuda上

        mention_hidden_size = self.model.config.hidden_size
        entity_encoding_hidden_size = self.precomputed_candidate_encoding.shape[1]
        self.mention_transform = nn.Linear(mention_hidden_size, entity_encoding_hidden_size) \
            if mention_hidden_size != entity_encoding_hidden_size else nn.Identity()

        self.train_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.eval_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        pass

    def entity_vocab_size(self):
        return self.precomputed_candidate_encoding.shape[0]

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        context: BertInput,
        candidate: BertInput,
        label: torch.Tensor,
        random_negs=True,
    ):

        embedding_ctxt = self.encode_context(context=context)
        embedding_ctxt = self.mention_transform(embedding_ctxt)

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        cand_encoding = self.precomputed_candidate_encoding
        # cand_encoding: [entity_num, emb_dim]
        if cand_encoding is not None:
            embedding_ctxt = embedding_ctxt.to(cand_encoding.device)
            scores = embedding_ctxt.mm(cand_encoding.t())
            # scores: [batch_size, entity_num]

            if self.training:
                loss = self.train_loss_fn(input=scores, target=label)
            else:
                loss = self.eval_loss_fn(input=scores, target=label)
            # loss = torch.tensor(np.nan)
        else:
            # Train time. We compare with all elements of the batch
            _, embedding_cands = self.model(
                # context
                token_idx_ctxt=None,
                segment_idx_ctxt=None,
                mask_ctxt=None,
                # candidate
                token_idx_cands=candidate.input_ids,
                segment_idx_cands=candidate.token_type_ids,
                mask_cands=candidate.attention_mask,
            )
            if random_negs:
                # train on random negatives
                scores = embedding_ctxt.mm(embedding_cands.t())
                loss = None
            else:
                # train on hard negatives  # 给每个mention喂一个样本
                embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
                embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 1
                scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
                scores = torch.squeeze(scores)
                loss = None
        return scores, loss

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, batch: ModelExample):
        random_negs = batch.label is None

        if self.candidate_encoding_device is None:
            if self.precomputed_candidate_encoding.device != self.device:
                self.precomputed_candidate_encoding = self.precomputed_candidate_encoding.to(self.device)
        else:
            if self.precomputed_candidate_encoding.device != self.candidate_encoding_device:
                self.precomputed_candidate_encoding = self.precomputed_candidate_encoding.to(self.candidate_encoding_device)


        scores, loss = self.score_candidate(context=batch.q_text, candidate=batch.c_text, random_negs=random_negs, label=batch.label)
        # scores: [batch_size, entity_num]

        topk_values, topk_indices = scores.topk(k=self.out_topk_candidates, dim=1)
        # topk_values, topk_indices: [batch_size, out_top_k]
        result = BaseModelReturn(
            predict_score=scores,
            out_topk_predict_score=topk_values,
            out_topk_indices=topk_indices,
            loss=loss,
        )
        return result


if __name__ == '__main__':
    BlinkBiEncoderRanker(params=...)

