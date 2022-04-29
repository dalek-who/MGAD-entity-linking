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
from global_link_models.luke_link_local import EntityPredictionHeadTransform
from emb_distill_models.model_type_hints import BaseModelReturn as Distill_BaseModelReturn
from emb_distill_dataset import ModelExample as Distill_ModelExample
from global_link_dataset import WordBertInput, EntityBertInput, EntityMLMLabel, WordMLMLabel
from global_link_models.model_type_hints import BaseModelReturn as GlobalLink_BaseModelReturn
import utils.special_tokens as st


def get_luke(
        bert_pretrained_dir: Union[Path, str],
        luke_pretrained_dir: Union[Path, str],
        entity_emb_size: int,
        entity_vocab_size: int,
):
    bert_pretrained_dir, luke_pretrained_dir = Path(bert_pretrained_dir), Path(luke_pretrained_dir)
    assert re.split(r"-|_", bert_pretrained_dir.name)[-1] == re.split(r"-|_", luke_pretrained_dir.name)[-1], \
        (bert_pretrained_dir, luke_pretrained_dir)

    bert_config = AutoConfig.from_pretrained(bert_pretrained_dir)
    bert_model_name = bert_pretrained_dir.name if isinstance(bert_pretrained_dir, Path) \
        else bert_pretrained_dir.split("/")[-1]
    luke_config = LukeConfig(
        entity_vocab_size=entity_vocab_size,
        bert_model_name=bert_model_name,
        entity_emb_size=entity_emb_size,
        **bert_config.to_dict(),
    )
    luke = LukeModel(config=luke_config)
    luke_state_dict = torch.load(Path(luke_pretrained_dir) / "pytorch_model.bin", map_location="cpu")
    entity_emb_key = 'entity_embeddings.entity_embeddings.weight'
    luke_state_dict[entity_emb_key] = luke_state_dict[entity_emb_key][:entity_vocab_size]
    luke.load_state_dict(luke_state_dict, strict=False)

    transform = EntityPredictionHeadTransform(config=luke_config)
    transform_state_dict = {
        name: luke_state_dict[f"entity_predictions.transform.{name}"]
        for name in transform.state_dict().keys()
    }
    transform.load_state_dict(transform_state_dict, strict=True)
    return luke, transform


# 包含模型的所有参数和必要的操作，但是不允许用来forward
# 根据具体任务选择下面的子类之一用来forward
class EntityDescriptionLukeBase(pl.LightningModule):
    allow_manually_init: bool = False  # 是否允许在初始化时人工指定context_init_checkpoint、entity_init_checkpoint
    # 注：若不人工指定，则默认初始化为luke或bert

    def __init__(
            self,
            py_logger: Logger,

            bert_pretrained_dir: Union[Path, str],
            luke_pretrained_dir: Union[Path, str],

            context_init_checkpoint: Union[Path, str, None],
            entity_init_checkpoint: Union[Path, str, None],

            share_backbone: bool,
            entity_encoder_use_luke: bool,
            special_entity_emb_size: int,
            entity_pooling: Literal["average", "cls", "luke_span"],
            do_transform: bool,

            *args,
            **kwargs,
    ):
        super().__init__()
        self.py_logger = py_logger
        self.entity_pooling = entity_pooling
        self.do_transform = do_transform

        entity_emb_key = 'entity_embeddings.entity_embeddings.weight'

        # context encoder
        self.context_encoder, context_transform = get_luke(
            bert_pretrained_dir=bert_pretrained_dir,
            luke_pretrained_dir=luke_pretrained_dir,
            entity_emb_size=special_entity_emb_size,
            entity_vocab_size=4,  # 只保留前4个特殊token
        )
        if do_transform:
            self.context_transform = context_transform
        else:
            self.context_transform = None

        # load init weight
        if self.allow_manually_init:
            if context_init_checkpoint is not None:
                context_init_checkpoint = Path(context_init_checkpoint)
                context_state_dict = torch.load(context_init_checkpoint, map_location="cpu")
                context_state_dict = context_state_dict if "state_dict" in context_state_dict else context_state_dict
                context_state_dict[entity_emb_key] = context_state_dict[entity_emb_key][:4]
                self.context_encoder.load_state_dict(context_state_dict, strict=False)
                if self.context_transform is not None:
                    context_transform_state_dict = {
                        name: context_state_dict[f"entity_predictions.transform.{name}"]
                        for name in self.context_transform.state_dict().keys()
                    }
                    self.context_transform.load_state_dict(context_transform_state_dict, strict=True)
                self.py_logger.info(f"load context checkpoint: {context_init_checkpoint.absolute()}")
        else:
            assert context_init_checkpoint is None

        # entity encoder
        self.share_backbone = share_backbone
        if share_backbone:  # share backbone
            assert entity_encoder_use_luke
            assert context_init_checkpoint == entity_init_checkpoint
            self.candidate_encoder = self.context_encoder
            self.candidate_transform = self.context_transform
            self.py_logger.info(f"entity checkpoint: share")
        else:
            if entity_encoder_use_luke:  # two backbone, same init weight from luke
                self.candidate_encoder, candidate_transform = get_luke(
                    bert_pretrained_dir=bert_pretrained_dir,
                    luke_pretrained_dir=luke_pretrained_dir,
                    entity_emb_size=special_entity_emb_size,
                    entity_vocab_size=4,
                )
                if do_transform:
                    self.candidate_transform = candidate_transform
                else:
                    self.candidate_transform = None
            else:   # two backbone, mention use luke, entity use bert
                self.candidate_encoder = AutoModel.from_pretrained(bert_pretrained_dir)
                self.candidate_transform = None

            # load init weight
            if self.allow_manually_init:
                if entity_init_checkpoint is not None:
                    entity_init_checkpoint = Path(entity_init_checkpoint)
                    entity_state_dict = torch.load(entity_init_checkpoint, map_location="cpu")
                    entity_state_dict = entity_state_dict if "state_dict" in entity_state_dict else entity_state_dict
                    self.candidate_encoder.load_state_dict(entity_state_dict, strict=True)
                    if self.candidate_transform is not None:
                        candidate_transform_state_dict = {
                            name: entity_state_dict[f"entity_predictions.transform.{name}"]
                            for name in self.candidate_transform.state_dict().keys()
                        }
                        self.candidate_transform.load_state_dict(candidate_transform_state_dict, strict=True)
                    self.py_logger.info(f"load entity checkpoint: {entity_init_checkpoint.absolute()}")
            else:
                assert entity_init_checkpoint is None

        if self.do_transform:
            assert self.context_transform is not None
            assert self.candidate_transform is not None
        else:
            assert self.context_transform is None
            assert self.candidate_transform is None

        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please use a child model instead")

    def encode_entity_description(self, entity_text: WordBertInput):
        batch_size = entity_text.input_ids.shape[0]
        # encode
        if isinstance(self.candidate_encoder, LukeModel):
            if self.entity_pooling == "luke_span":
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

                word_seq_output, entity_seq_output, pooled_output = self.candidate_encoder(
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
                word_seq_output, pooled_output = self.candidate_encoder(
                    word_ids=entity_text.input_ids,
                    word_segment_ids=entity_text.token_type_ids,
                    word_attention_mask=entity_text.attention_mask,
                )  # [entity_num,  seq_len, hidden_size]
                luke_full_seq_emb = None
            cls = word_seq_output[:, 0, :]  # [entity_num, hidden_size]
        else:  # bert / roberta
            assert self.entity_pooling != "luke_span"
            bert_return: BaseModelOutputWithPoolingAndCrossAttentions = self.candidate_encoder(
                input_ids=entity_text.input_ids,
                token_type_ids=entity_text.token_type_ids,
                attention_mask=entity_text.attention_mask,
            )
            last_hidden_state = bert_return.last_hidden_state   # [entity_num,  seq_len, hidden_size]
            cls = last_hidden_state[:, 0, :]  # [entity_num, hidden_size]
            luke_full_seq_emb = None

        # pooling
        if self.entity_pooling == "average":
            entity_emb = self.average_pooling(
                sequence_hidden_state=word_seq_output, attention_mask=entity_text.attention_mask)
        elif self.entity_pooling == "cls":
            entity_emb = cls
        elif self.entity_pooling == "luke_span":
            entity_emb = luke_full_seq_emb
        else:
            raise ValueError(self.entity_pooling)

        if self.do_transform:
            entity_emb = self.candidate_transform(entity_emb)
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

    def encode_context_word_entity_sequence(self, word_seq: WordBertInput, entity_seq: EntityBertInput):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility
        device = next(iter(self.parameters())).device

        # 注：
        # entity_ids被mask的实体值为2（[mask]的index)，其他位置是实体vocab_id，剩下的补位的是0
        # masked_entity_labels对应被mask的实体(entity_ids里值为2的位置）值为vocab_id,剩下位置为-1。entity没有word的替换、不变策略，只有mask
        # (masked_entity_labels>=0) == (entity_ids==2)
        # masked_lm_labels 除了被mask的东西都是-1，其label比word_ids为mask（50264）的多，因为除了被mask的词，还有被替换的、保持不变的（bert策略）
        # entity_segment_ids这里全是0，entity_attention_mask关心的实体是1，后面补位的pad是0（和entity_ids是对应的，后面补位的pad是0）
        # 例: entity_position_ids[i][j] = [119, 120, 121, 122, 123,  -1,  -1,  -1,  -1,  -1,  -1]


        # word_ids, word_segment_ids, word_attention_mask: [batch_size, context_seq_len]
        # entity_ids, entity_segment_ids, entity_attention_mask:  [batch_size, batch_max_entity_num]
        # entity_position_ids: [batch_size, max_entity_num, max_span_len]
        output = self.context_encoder(
            word_ids=word_seq.input_ids,
            word_segment_ids=word_seq.token_type_ids,
            word_attention_mask=word_seq.attention_mask,

            entity_ids=entity_seq.input_ids,
            entity_position_ids=entity_seq.position_ids,
            entity_segment_ids=entity_seq.token_type_ids,
            entity_attention_mask=entity_seq.attention_mask,
        )

        # encode context sequence and additional (masked) entity sequence
        word_sequence_output, entity_sequence_output = output[:2]
        # word_sequence_output: [batch_size, context_seq_len, hidden_size]
        # entity_sequence_output: [batch_size, max_entity_num, hidden_size]

        if self.do_transform:
            entity_sequence_output = self.context_transform(entity_sequence_output)

        return word_sequence_output, entity_sequence_output


class EntityDescriptionLukeForTrain(EntityDescriptionLukeBase):
    allow_manually_init = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # warning: 子类中不要新增含权重的layer
        self.train_loss_fn = CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.eval_loss_fn = CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def forward(
            self,
            context_word_seq: WordBertInput,
            context_entity_seq: EntityBertInput,
            candidate_word_seq: WordBertInput,

            masked_entity_label: EntityMLMLabel,
            unique_entity_label_value: torch.Tensor,
    ):

        ######################################
        #   encode context and mention
        ######################################

        # # # encode

        ctxt_word_seq_output, ctxt_entity_seq_output = self.encode_context_word_entity_sequence(
            word_seq=context_word_seq, entity_seq=context_entity_seq)
        # ctxt_word_seq_output: [batch_size, word_seq_len, hidden_size]
        # ctxt_entity_seq_output: [batch_size, max_entity_num, hidden_size]

        # # # select masked entity

        # entity_mask: [batch_size, max_entity_num]
        target_ctxt_entity_seq_output = torch.masked_select(
            ctxt_entity_seq_output, masked_entity_label.mlm_mask.bool().unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_ctxt_entity_seq_output: [batch_mask_entity_num * hidden_size]
        context_hidden_size = ctxt_entity_seq_output.shape[-1]
        target_ctxt_entity_seq_output = target_ctxt_entity_seq_output.view(-1, context_hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ######################################
        #   encode entity description
        ######################################

        cand_entity_emb = self.encode_entity_description(entity_text=candidate_word_seq)
        # [cand_num, hidden_size]

        ######################################
        #   compute match score
        ######################################

        in_batch_score: torch.Tensor = torch.einsum(
            "b h, c h -> b c",
            target_ctxt_entity_seq_output, cand_entity_emb
        )
        # [batch_mask_entity_num, cand_num]

        ######################################
        #   compute loss
        ######################################

        # # # convert to local index for compute loss

        target_entity_labels = torch.masked_select(
            masked_entity_label.mlm_label, masked_entity_label.mlm_mask.bool())
        # [batch_mask_entity_num]

        assert unique_entity_label_value.shape[0] == candidate_word_seq.input_ids.shape[0]
        target_entity_labels_local_index = torch.tensor([
            unique_entity_label_value.tolist().index(label)
            for label in target_entity_labels
        ], device=self.device)  # [batch_mask_entity_num]

        # # # compute loss

        if self.training:
            loss = self.train_loss_fn(input=in_batch_score, target=target_entity_labels_local_index)
        else:
            loss = self.eval_loss_fn(input=in_batch_score, target=target_entity_labels_local_index)

        ###########################################
        #   get topk
        ###########################################

        topk_scores, topk_local_index = in_batch_score.topk(k=in_batch_score.shape[1], dim=1)
        # topk_scores, topk_local_index: [batch_mask_entity_num, out_top_k]

        topk_labels = torch.stack([
            unique_entity_label_value[topk_local_index[i]]
            for i in range(topk_local_index.shape[0])
        ])

        ###########################################
        #   prepare return
        ###########################################

        model_return = GlobalLink_BaseModelReturn(
            # necessary
            label=target_entity_labels,
            predict_score=in_batch_score,
            out_topk_predict_score=topk_scores,
            out_topk_indices=topk_labels,
            loss=loss,
            mention_emb=target_ctxt_entity_seq_output,
            entity_emb=cand_entity_emb,

            # unnecessary
            query_id=...,
            freq_bin=...,
            gold_wikidata_id=...,
            gold_wikipedia_pageid=...,
            has_gold=...,
        )

        return model_return


class EntityDescriptionLukeForMentionEncode(EntityDescriptionLukeBase):
    allow_manually_init = False

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            seed: int,

            *args,
            **kwargs
    ):
        super().__init__(
            context_init_checkpoint=None,
            entity_init_checkpoint=None,
            *args, **kwargs
        )
        self.tokenizer = tokenizer
        self.seed = seed
        # warning: 子类中不要新增含权重的layer

    def forward(self, batch: Distill_ModelExample):
        ctxt_word_seq_output, ctxt_entity_seq_output = self.encode_context_word_entity_sequence(
            word_seq=batch.luke_context_word_seq,
            entity_seq=batch.luke_context_entity_seq,
        )
        # ctxt_word_seq_output: [batch_size, word_seq_len, hidden_size]
        # ctxt_entity_seq_output: [batch_size, max_entity_num, hidden_size]

        # # # select masked entity

        entity_mask = (batch.luke_context_entity_seq.attention_mask >= 0)
        # entity_mask: [batch_size, max_entity_num]
        target_ctxt_entity_seq_output = torch.masked_select(
            ctxt_entity_seq_output, entity_mask.unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_ctxt_entity_seq_output: [batch_mask_entity_num * hidden_size]
        context_hidden_size = ctxt_entity_seq_output.shape[-1]
        target_ctxt_entity_seq_output = target_ctxt_entity_seq_output.view(-1, context_hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ###########################################
        #   prepare return
        ###########################################

        model_return = Distill_BaseModelReturn(
            student_mention_emb=target_ctxt_entity_seq_output,

            loss_emb=torch.tensor(np.nan, device=self.device),
            loss_bias=torch.tensor(np.nan, device=self.device),
            loss_total=torch.tensor(np.nan, device=self.device),
            loss_student_link=torch.tensor(np.nan, device=self.device),
            loss_score_distill=torch.tensor(np.nan, device=self.device),

            student_entity_emb=None,
            student_entity_bias=None,
            wikipedia_pageid=batch.wikipedia_pageid,
            batch_size=target_ctxt_entity_seq_output.shape[0],
        )

        return model_return

    def entity_vocab_size(self):
        return None


class EntityDescriptionLukeForEntityEncode(EntityDescriptionLukeBase):
    allow_manually_init = False

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            seed: int,

            *args,
            **kwargs
    ):
        super().__init__(
            context_init_checkpoint=None,
            entity_init_checkpoint=None,
            *args,
            **kwargs
        )
        self.tokenizer = tokenizer
        self.seed = seed
        # warning: 子类中不要新增含权重的layer

    def forward(self, batch: Distill_ModelExample):
        entity_emb = self.encode_entity_description(entity_text=batch.entity_text)

        model_return = Distill_BaseModelReturn(
            loss_emb_ce=torch.tensor(np.nan, device=self.device),
            loss_emb_mse=torch.tensor(np.nan, device=self.device),
            loss_emb_invert_ce=torch.tensor(np.nan, device=self.device),
            loss_bias=torch.tensor(np.nan, device=self.device),
            loss_total=torch.tensor(np.nan, device=self.device),
            loss_student_link=torch.tensor(np.nan, device=self.device),
            loss_score_distill=torch.tensor(np.nan, device=self.device),

            student_entity_emb=entity_emb,
            student_entity_bias=None,
            wikipedia_pageid=batch.wikipedia_pageid,
            batch_size=entity_emb.shape[0],
        )
        return model_return

    def entity_vocab_size(self):
        return None


class EntityDescriptionLukeForLink(EntityDescriptionLukeBase):
    allow_manually_init = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # warning: 子类中不要新增含权重的layer

    def forward(self, *args, **kwargs):
        ...
