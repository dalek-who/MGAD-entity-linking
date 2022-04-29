import random
from typing import Optional, List, Tuple, Dict, NamedTuple, Union
from typing_extensions import Literal
from pathlib import Path
from logging import Logger
from itertools import chain
import re

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
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

from luke_model.luke_model import LukeModel, LukeConfig
from global_link_models.luke_link_local import LukeLinkLocal, EntityPredictionHeadTransform
from global_link_models.model_type_hints import BaseModelReturn as GlobalLink_BaseModelReturn
from emb_distill_dataset import ModelExample as Distill_ModelExample, WordBertInput, EntityBertInput
from global_link_dataset import ModelExample as GlobalLink_ModelExample, \
    EntityMLMLabel as GlobalLink_EntityMLMLabel
from emb_distill_models.model_type_hints import BaseModelReturn as Distill_BaseModelReturn
from emb_distill_models.entity_discription_luke import EntityDescriptionLukeForTrain, EntityDescriptionLukeForEntityEncode, get_luke
from emb_distill_models.luke_teacher_student import EmbDistillHead
import utils.special_tokens as st
from utils.freeze_model import freeze, un_freeze


def extract_state_dict(checkpoint_path: Union[Path, str]):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {re.sub(r"^model\.", "", k): v for k,v in state_dict.items()}
    return state_dict


class EmbDistillTeacherStudent(pl.LightningModule):
    def __init__(
            self,
            bert_pretrained_dir: Path,
            luke_pretrained_dir: Path,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            seed: int,
            py_logger: Logger,

            emb_distill_ce_loss_weight: float,  # embedding蒸馏的CrossEntropy loss占比
            emb_distill_mse_loss_weight: float,  # embedding蒸馏的mse loss占比

            # teacher specific
            teacher_strict: bool,
            teacher_entity_emb_size: int,
            teacher_entity_vocab_size: int,
            teacher_use_entity_bias: bool,
            teacher_init_checkpoint: Union[Path, str],

            # student specific
            student_use_luke: bool,
            student_entity_pooling: str,

            *args,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.py_logger = py_logger

        self.student_entity_pooling = student_entity_pooling

        # loss
        assert 0 <= emb_distill_ce_loss_weight <= 1, emb_distill_ce_loss_weight
        assert 0 <= emb_distill_mse_loss_weight <= 1, emb_distill_mse_loss_weight
        assert emb_distill_ce_loss_weight + emb_distill_mse_loss_weight == 1, \
            (emb_distill_ce_loss_weight, emb_distill_mse_loss_weight)

        self.emb_distill_ce_loss_weight = emb_distill_ce_loss_weight
        self.emb_distill_mse_loss_weight = emb_distill_mse_loss_weight

        # teacher
        teacher = LukeLinkLocal(
            # un-necessary
            in_topk_candidates=...,
            max_context_len=...,
            max_entity_len=...,
            max_span_len=...,
            mention_add_start_end=...,
            model_data_format=...,
            entity_vocab_table_name=...,
            allow_nil=...,
            out_topk_candidates=...,

            # necessary
            pretrained_dir=bert_pretrained_dir,
            entity_vocab_size=teacher_entity_vocab_size,
            entity_emb_size=teacher_entity_emb_size,
            tokenizer=tokenizer,
            seed=seed,
            py_logger=py_logger,
            freeze_bert=True,
            freeze_entity=True,
            freeze_entity_prediction=True,
            use_entity_bias=teacher_use_entity_bias,
            candidate_from="list",
        )
        teacher_state_dict = extract_state_dict(checkpoint_path=teacher_init_checkpoint)
        teacher.load_state_dict(teacher_state_dict, strict=teacher_strict)
        freeze(teacher)

        self.teacher_emb_distill_head = EmbDistillHead(
            teacher_transform_layer=teacher.entity_predictions.transform,
            teacher_entity_emb_decoder=teacher.entity_predictions.decoder,
            teacher_entity_emb_bias=teacher.entity_predictions.bias,
        )
        assert not any([p.requires_grad for name, p in self.teacher_emb_distill_head.named_parameters()])

        # student
        if student_use_luke:
            self.student, transform = get_luke(
                bert_pretrained_dir=bert_pretrained_dir,
                luke_pretrained_dir=luke_pretrained_dir,
                entity_emb_size=teacher_entity_emb_size,
                entity_vocab_size=4,  # 只保留前4个特殊token
            )
        else:
            self.student = AutoModel.from_pretrained(bert_pretrained_dir)

    def forward(self, batch: Distill_ModelExample) -> Distill_BaseModelReturn:
        batch_size = batch.entity_text.input_ids.shape[0]

        student_entity_emb = self.encode_entity_description(entity_text=batch.entity_text)
        emb_distill_ce_loss, emb_distill_mse_loss, pred_classify_top1_score, pred_classify_top1_index = self.teacher_emb_distill_head(
            entity_emb=student_entity_emb, entity_emb_index=batch.teacher_emb_index
        )

        ######################################
        #   loss
        ######################################

        loss_total = torch.tensor(0., dtype=emb_distill_ce_loss.dtype, device=self.device)
        loss_total += emb_distill_ce_loss * self.emb_distill_ce_loss_weight
        loss_total += emb_distill_mse_loss * self.emb_distill_mse_loss_weight

        student_entity_bias=torch.zeros(student_entity_emb.shape[0], dtype=student_entity_emb.dtype, device=self.device)
        model_return = Distill_BaseModelReturn(
            batch_size=batch_size,
            loss_total=loss_total,

            loss_student_link=torch.tensor(np.nan, device=self.device),
            loss_score_distill=torch.tensor(np.nan, device=self.device),
            loss_bias=torch.tensor(np.nan, device=self.device),
            loss_emb_ce=emb_distill_ce_loss,
            loss_emb_mse=emb_distill_mse_loss,

            wikipedia_pageid=batch.wikipedia_pageid,
            student_entity_emb=student_entity_emb,
            student_entity_bias=student_entity_bias,

            student_label=batch.teacher_emb_index,
            student_predict_score=None,
            student_out_topk_predict_score=pred_classify_top1_score,
            student_out_topk_indices=pred_classify_top1_index,
        )
        return model_return

    def encode_entity_description(self, entity_text: WordBertInput):
        batch_size = entity_text.input_ids.shape[0]
        # encode
        if isinstance(self.student, LukeModel):
            if self.student_entity_pooling == "luke_span":
                entity_ids = torch.ones(size=(batch_size, 1), dtype=torch.long, device=self.device) * st.ENTITY_MASK_INDEX  # [batch_size, 1]
                entity_segment_ids = torch.zeros_like(entity_ids)  # [batch_size, 1]
                entity_attention_mask = torch.ones_like(entity_ids)  # [batch_size, 1]

                max_seq_len = entity_text.input_ids.shape[1]
                entity_position_ids = eop.repeat(  # [batch_size, 1, max_seq_len]
                    torch.arange(max_seq_len, device=self.device),
                    "seq_len -> batch_size span_num seq_len",
                    batch_size=batch_size, span_num=1
                )
                entity_position_ids[
                    ~eop.rearrange(entity_text.attention_mask, "batch seq -> batch 1 seq").bool()
                ] = -1

                word_seq_output, entity_seq_output, pooled_output = self.student(
                    word_ids=entity_text.input_ids,
                    word_segment_ids=entity_text.token_type_ids,
                    word_attention_mask=entity_text.attention_mask,

                    entity_ids=entity_ids,
                    entity_segment_ids=entity_segment_ids,
                    entity_attention_mask=entity_attention_mask,
                    entity_position_ids=entity_position_ids,
                )
                luke_full_seq_emb = entity_seq_output[:, 0, :]
                # word_seq_output: [batch_size, word_seq_len, hidden_size]
                # entity_seq_output: [batch_size, 1, hidden_size]
            else:
                word_seq_output, pooled_output = self.student(
                    word_ids=entity_text.input_ids,
                    word_segment_ids=entity_text.token_type_ids,
                    word_attention_mask=entity_text.attention_mask,
                )  # [entity_num,  seq_len, hidden_size]
                luke_full_seq_emb = None
            cls = word_seq_output[:, 0, :]  # [entity_num, hidden_size]
        else:  # bert / roberta
            assert self.student_entity_pooling != "luke_span"
            bert_return: BaseModelOutputWithPoolingAndCrossAttentions = self.student(
                input_ids=entity_text.input_ids,
                token_type_ids=entity_text.token_type_ids,
                attention_mask=entity_text.attention_mask,
            )
            last_hidden_state = bert_return.last_hidden_state   # [entity_num,  seq_len, hidden_size]
            word_seq_output = last_hidden_state
            cls = last_hidden_state[:, 0, :]  # [entity_num, hidden_size]
            luke_full_seq_emb = None

        # pooling
        if self.student_entity_pooling == "average":
            entity_emb = self.average_pooling(
                sequence_hidden_state=word_seq_output, attention_mask=entity_text.attention_mask)
        elif self.student_entity_pooling == "cls":
            entity_emb = cls
        elif self.student_entity_pooling == "luke_span":
            entity_emb = luke_full_seq_emb
        else:
            raise ValueError(self.student_entity_pooling)

        return entity_emb

    def average_pooling(self, sequence_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        # sequence_hidden_state: [batch_size, seq_len, hidden_state]
        # attention_mask: [batch_size, seq_len]

        assert sequence_hidden_state.shape[:2] == attention_mask.shape
        pooled = torch.stack([
            eop.reduce(
                sequence_hidden_state[i][attention_mask[i]],
                pattern="s h -> h", reduction="mean"
            )
            for i in range(sequence_hidden_state.shape[0])
        ], dim=0)  # [batch_size, hidden_state]
        return pooled

    def entity_vocab_size(self):
        return self.teacher_emb_distill_head.entity_emb.weight.shape[0]


