from typing import Optional, List, Tuple, Dict, NamedTuple, Union
from pathlib import Path
from logging import Logger
from itertools import chain
import re

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer, PretrainedConfig
from torch.nn import LayerNorm as BertLayerNorm
from transformers.models.bert import BertModel
from transformers.activations import ACT2FN

import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device

from luke_model.luke_model import LukeModel, LukeConfig
from emb_distill_dataset import ModelExample
from emb_distill_models.model_type_hints import BaseModelReturn


"""
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


luke：
class EntityPredictionHead(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityPredictionHead, self).__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)  # 全连接，激活，LayerNorm
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states

class RobertaLMHead(nn.Module):
    '''Roberta Head for masked language modeling.'''

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

"""


class TransformerLinearBlock(nn.Module):
    """
    BertIntermediate and BertOutput
    equals to:
        from transformers.models.bert.modeling_bert import BertOutput, BertIntermediate
    linear, activate, linear, dropout, add & norm
    """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            layer_norm_eps: float,
            hidden_dropout_prob: float,
    ):
        # BertIntermediate
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        # BertOutput
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x):
        # x: [batch_size, hidden_size]
        # BertIntermediate
        hidden_states = self.dense_1(x)  # [batch_size, intermediate_size]
        hidden_states = self.intermediate_act_fn(hidden_states)  # [batch_size, intermediate_size]

        # BertOutput
        hidden_states = self.dense_2(hidden_states)  # [batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)  # [batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states + x)  # [batch_size, hidden_size]
        return hidden_states


class FeatureTransform(pl.LightningModule):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            do_activate: bool,
            do_layer_norm: bool,
            do_dropout: bool,
            dropout: float,
            num_layer: int,
            layer_norm_eps: float,
    ):
        super().__init__()
        layers = tuple(chain.from_iterable([
            (
                nn.Linear(in_features=in_dim, out_features=out_dim) if i==0 else nn.Linear(in_features=out_dim, out_features=out_dim),
                BertLayerNorm(normalized_shape=out_dim) if do_layer_norm else nn.Identity(),
                nn.GELU() if do_activate else nn.Identity(),
                nn.Dropout(dropout) if do_dropout else nn.Identity(),
            )
            for i in range(num_layer)
        ]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class Feature2Emb(pl.LightningModule):
    def __init__(
            self,
            bert_pretrained_dir: Union[str, Path],
            luke_pretrained_dir: Union[str, Path],
            entity_vocab_size: int,
            entity_emb_size: int,
            tokenized_entity_lmdb_dir: Union[Path, str],
            tokenized_link_dataset_lmdb_dir: Union[Path, str],
            feature_do_activate: bool,
            feature_do_dropout: bool,
            feature_do_layer_norm: bool,
            feature_num_layer: int,
            teacher_entity_vocab_table_name: str,
            max_seq_len: int,
            max_span_len: int,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            seed: int,
            py_logger: Logger,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.loss_function = nn.MSELoss(reduction="mean")

        bert_pretrained_dir, luke_pretrained_dir = Path(bert_pretrained_dir), Path(luke_pretrained_dir)
        assert re.split(r"-|_", bert_pretrained_dir.name)[-1] == re.split(r"-|_", luke_pretrained_dir.name)[-1], \
            (bert_pretrained_dir, luke_pretrained_dir)
        bert_config = AutoConfig.from_pretrained(bert_pretrained_dir)
        bert_model_name = bert_pretrained_dir.name if isinstance(bert_pretrained_dir, Path) else bert_pretrained_dir.split("/")[-1]
        config = LukeConfig(
            entity_vocab_size=entity_vocab_size,
            bert_model_name=bert_model_name,
            entity_emb_size=entity_emb_size,
            **bert_config.to_dict(),
        )
        self.backbone: nn.Module = LukeModel(config)
        luke_state_dict = torch.load(Path(luke_pretrained_dir) / "pytorch_model.bin")
        self.backbone.load_state_dict(state_dict=luke_state_dict, strict=False)

        self.feature_transform = FeatureTransform(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size,
            dropout=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            do_activate=feature_do_activate,
            do_dropout=feature_do_dropout,
            do_layer_norm=feature_do_layer_norm,
            num_layer=feature_num_layer,
        )
        self.emb_transform = nn.Linear(in_features=config.hidden_size, out_features=config.entity_emb_size)
        self.bias_transform = nn.Linear(in_features=config.hidden_size, out_features=1)

    def forward(self, batch: ModelExample) -> BaseModelReturn:
        word_sequence_output, pooled_output = self.backbone(
            word_ids=batch.entity_text.input_ids,
            word_segment_ids=batch.entity_text.token_type_ids,
            word_attention_mask=batch.entity_text.attention_mask,
        )
        bert_cls = word_sequence_output[:, 0, :]
        transformed_feature = self.feature_transform(bert_cls)
        student_entity_emb = self.emb_transform(transformed_feature)
        student_entity_bias = self.bias_transform(transformed_feature)

        loss_emb = self.loss_function(input=student_entity_emb, target=batch.teacher_emb) \
            if batch.teacher_emb is not None else torch.tensor(np.nan)
        loss_bias = self.loss_function(input=student_entity_bias, target=batch.teacher_bias.view_as(student_entity_bias)) \
            if batch.teacher_bias is not None else torch.tensor(np.nan)
        loss_total = loss_emb + loss_bias

        model_return = BaseModelReturn(
            loss_emb=loss_emb,
            loss_bias=loss_bias,
            loss_total=loss_total,
            student_entity_emb=student_entity_emb,
            student_entity_bias=student_entity_bias,
            wikipedia_pageid=batch.wikipedia_pageid,
            batch_size=student_entity_emb.shape[0],

            loss_student_link=torch.tensor(np.nan, device=loss_total.device),
            loss_score_distill=torch.tensor(np.nan, device=loss_total.device),
        )
        return model_return

    def entity_vocab_size(self):
        return None

