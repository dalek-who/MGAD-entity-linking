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


from link_models.blink_biencoder import BlinkBiEncoderBase
from link_dataset import BertInput
from emb_distill_models.model_type_hints import BaseModelReturn as Distill_BaseModelReturn
from emb_distill_dataset import ModelExample


class BlinkBiEncoderForMentionEncode(BlinkBiEncoderBase):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast,
            py_logger: logging.Logger=None,
            *args,
            **kwargs
    ):
        super().__init__(
            have_ctxt_encoder=True, have_cand_encoder=False,
            *args, **kwargs
        )
        self.tokenizer = tokenizer
        self.py_logger = py_logger
        pass

    def entity_vocab_size(self):
        return None

    def forward(self, batch: ModelExample):
        context = BertInput(
            input_ids=batch.luke_context_word_seq.input_ids,
            token_type_ids=batch.luke_context_word_seq.token_type_ids,
            attention_mask=batch.luke_context_word_seq.attention_mask,
        )
        embedding_ctxt = self.encode_context(context=context)

        model_return = Distill_BaseModelReturn(
            student_mention_emb=embedding_ctxt,

            loss_emb=torch.tensor(np.nan, device=self.device),
            loss_bias=torch.tensor(np.nan, device=self.device),
            loss_total=torch.tensor(np.nan, device=self.device),
            loss_student_link=torch.tensor(np.nan, device=self.device),
            loss_score_distill=torch.tensor(np.nan, device=self.device),

            student_entity_emb=None,
            student_entity_bias=None,
            wikipedia_pageid=batch.wikipedia_pageid,
            batch_size=embedding_ctxt.shape[0],
        )
        return model_return


