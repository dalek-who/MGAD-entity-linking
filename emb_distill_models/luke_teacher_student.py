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
from emb_distill_models.entity_discription_luke import EntityDescriptionLukeForTrain

from utils.freeze_model import freeze, un_freeze


# 唯一值unique的每个元素，在values中第一次出现的index
def get_first_occur(unique_value: torch.Tensor, values: torch.Tensor):
    assert torch.unique(unique_value).shape == unique_value.shape
    first_occur = torch.cat([torch.nonzero(values==v)[0] for v in unique_value])  # [len_unique_values]
    return first_occur


# values中的每个元素，在unique中的位置，相当于get_first_occur的对偶操作
def get_unique_value_index(unique_value: torch.Tensor, values: torch.Tensor):
    assert torch.unique(unique_value).shape == unique_value.shape
    unique_value_index = torch.cat([torch.nonzero(unique_value==v)[0] for v in values])  # [len_values]
    return unique_value_index


class EmbDistillHead(pl.LightningModule):
    def __init__(
            self,
            teacher_transform_layer: Union[EntityPredictionHeadTransform, nn.Identity],
            teacher_entity_emb_decoder: Union[nn.Linear, nn.Embedding],
            teacher_entity_emb_bias: Union[nn.Parameter, torch.Tensor],
    ):
        super().__init__()
        self.transform = teacher_transform_layer

        # cross entropy classifier
        entity_num, entity_emb_size = teacher_entity_emb_decoder.weight.shape[0], teacher_entity_emb_decoder.weight.shape[1]
        self.entity_decoder = nn.Linear(entity_emb_size, entity_num, bias=False)
        self.entity_decoder.weight = teacher_entity_emb_decoder.weight
        self.entity_decoder_bias = teacher_entity_emb_bias

        # mse loss embedding
        self.entity_emb = nn.Embedding(num_embeddings=entity_num, embedding_dim=entity_emb_size)
        self.entity_emb.weight = teacher_entity_emb_decoder.weight

        self.mse_fn = nn.MSELoss(reduction="mean")
        self.ce_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    def forward(self, entity_emb: torch.Tensor, entity_emb_index: torch.Tensor):
        transformed_entity_emb = self.transform(entity_emb)

        # cross entropy
        full_pred_score = self.entity_decoder(transformed_entity_emb) + self.entity_decoder_bias  # [batch_size, full_entity_num]
        ce_loss = self.ce_fn(input=full_pred_score, target=entity_emb_index)
        pred_classify_top1_score, pred_classify_top1_index = full_pred_score.topk(k=1, dim=-1)  # [batch_size, 1]

        teacher_emb = self.entity_emb(entity_emb_index)

        # mse
        mse_loss = self.mse_fn(input=transformed_entity_emb, target=teacher_emb)

        # invert cross entropy
        invert_pred_score: torch.Tensor = torch.einsum("t h, s h -> t s", teacher_emb, transformed_entity_emb)
        assert invert_pred_score.shape[0] == invert_pred_score.shape[1]
        invert_label = torch.arange(teacher_emb.shape[0]).to(invert_pred_score.device)
        invert_ce_loss = self.ce_fn(input=invert_pred_score, target=invert_label)

        return ce_loss, mse_loss, invert_ce_loss, pred_classify_top1_score, pred_classify_top1_index


class LukeTeacherStudent(pl.LightningModule):
    def __init__(
            self,
            bert_pretrained_dir: Path,
            luke_pretrained_dir: Path,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            seed: int,
            py_logger: Logger,

            distill_temperature: float,
            student_link_loss_weight: float,  # student link loss占比
            score_distill_loss_weight: float,  # score蒸馏loss占比
            emb_distill_ce_loss_weight: float,  # embedding蒸馏的CrossEntropy loss占比
            emb_distill_mse_loss_weight: float,  # embedding蒸馏的mse loss占比
            emb_distill_invert_ce_loss_weight: float,  # embedding蒸馏的逆向CrossEntropy loss占比
            do_transform: bool,

            # teacher specific
            teacher_entity_emb_size: int,
            teacher_entity_vocab_size: int,
            teacher_use_entity_bias: bool,
            teacher_out_topk_candidates: int,
            teacher_init_checkpoint: Union[Path, str],

            # student specific
            student_context_init_checkpoint: Union[Path, str],
            student_entity_init_checkpoint: Union[Path, str],
            student_entity_encoder_use_luke: bool,
            student_share_backbone: bool,
            student_entity_pooling: str,
            teacher_transform: bool,
            student_transform: bool,

            *args,
            **kwargs,
    ):
        super(LukeTeacherStudent, self).__init__()
        self.tokenizer = tokenizer
        self.py_logger = py_logger


        self.distill_temperature = distill_temperature

        # loss
        assert 0 <= student_link_loss_weight <= 1, student_link_loss_weight
        assert 0 <= score_distill_loss_weight <= 1, score_distill_loss_weight
        assert 0 <= emb_distill_ce_loss_weight <= 1, emb_distill_ce_loss_weight
        assert 0 <= emb_distill_mse_loss_weight <= 1, emb_distill_mse_loss_weight
        assert 0 <= emb_distill_invert_ce_loss_weight < 1, emb_distill_invert_ce_loss_weight
        assert student_link_loss_weight \
               + score_distill_loss_weight \
               + emb_distill_ce_loss_weight \
               + emb_distill_mse_loss_weight \
               + emb_distill_invert_ce_loss_weight == 1, \
            (student_link_loss_weight, score_distill_loss_weight, emb_distill_ce_loss_weight, emb_distill_mse_loss_weight, emb_distill_invert_ce_loss_weight)

        self.student_link_loss_weight = student_link_loss_weight
        self.score_distill_loss_weight = score_distill_loss_weight
        self.emb_distill_ce_loss_weight = emb_distill_ce_loss_weight
        self.emb_distill_mse_loss_weight = emb_distill_mse_loss_weight
        self.emb_distill_invert_ce_loss_weight = emb_distill_invert_ce_loss_weight

        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        if do_transform:
            assert teacher_transform ^ student_transform
        else:
            assert not teacher_transform and not student_transform

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
            out_topk_candidates=teacher_out_topk_candidates,
            candidate_from="list",
        )
        freeze(teacher)
        # teacher_state_dict = torch.load(Path(luke_pretrained_dir) / "pytorch_model.bin", map_location="cpu")
        # self.teacher.load_state_dict(teacher_state_dict, strict=False)

        teacher_checkpoint = torch.load(teacher_init_checkpoint, map_location="cpu")
        teacher_checkpoint = teacher_checkpoint["state_dict"] if "state_dict" in teacher_checkpoint else teacher_checkpoint
        teacher_checkpoint = {re.sub(r"^model\.", "", k): v for k,v in teacher_checkpoint.items()}
        teacher.load_state_dict(teacher_checkpoint)
        self.py_logger.info(f"load teacher checkpoint: {Path(teacher_init_checkpoint).absolute()}")

        self.teacher: Optional[LukeLinkLocal] = teacher if self.score_distill_loss_weight>0 else None

        # student
        self.student = EntityDescriptionLukeForTrain(  # luke作为backbone的权重在模型里面加载
            py_logger=py_logger,
            luke_pretrained_dir=luke_pretrained_dir,
            bert_pretrained_dir=bert_pretrained_dir,
            entity_encoder_use_luke=student_entity_encoder_use_luke,
            share_backbone=student_share_backbone,
            special_entity_emb_size=teacher_entity_emb_size,
            entity_pooling=student_entity_pooling,
            context_init_checkpoint=student_context_init_checkpoint,
            entity_init_checkpoint=student_entity_init_checkpoint,
            do_transform=student_transform,
        )

        if self.emb_distill_ce_loss_weight + self.emb_distill_mse_loss_weight + self.emb_distill_invert_ce_loss_weight > 0:
            if not do_transform:
                self.emb_distill_head = EmbDistillHead(
                    teacher_transform_layer=teacher.entity_predictions.transform,
                    teacher_entity_emb_decoder=teacher.entity_predictions.decoder,
                    teacher_entity_emb_bias=teacher.entity_predictions.bias,
                )
            elif do_transform and student_transform:
                self.emb_distill_head = EmbDistillHead(
                    teacher_transform_layer=nn.Identity(),
                    teacher_entity_emb_decoder=teacher.entity_predictions.decoder,
                    teacher_entity_emb_bias=teacher.entity_predictions.bias,
                )
            elif do_transform and teacher_transform:
                raise NotImplementedError()
            else:
                raise ValueError(do_transform, teacher_transform, student_transform)
            assert not any([p.requires_grad for name, p in self.emb_distill_head.named_parameters()])
        else:
            self.emb_distill_head = None


        pass

    def forward(self, batch: Distill_ModelExample) -> Distill_BaseModelReturn:
        # 解决in-batch的emb_index冲突：
        #   batch内的emb_index可能有重复值，获取其中的唯一值，并索引出对应的entity_text，以及in-batch label
        # example:
        #   emb_index: [20, 40, 40, 20, 50, 20, 30]
        #       ->  unique_value: [20, 40, 30, 50]
        #           first_occur: [0, 1, 6, 4]
        #           in_batch_label: [0, 1, 1, 0, 3, 0, 2]

        # batch.teacher_emb_index: [batch_size]
        if batch.teacher_emb_index is not None:
            teacher_emb_index = batch.teacher_emb_index
        else:
            teacher_emb_index = torch.masked_select(
                batch.luke_masked_entity_label.mlm_label, batch.luke_masked_entity_label.mlm_mask.bool()
            )
        unique_emb_index_value, inverse = torch.unique(teacher_emb_index, sorted=False, return_inverse=True)  # [unique_index_num]
        unique_emb_index_value: torch.Tensor
        assert len(teacher_emb_index) == batch.entity_text.input_ids.shape[0]

        # 未来除了in-batch实体外，还要加难负样本，所以不直接用inverse作为in_batch_label。未来会把gold emb_index

        batch_size = batch.luke_context_word_seq.input_ids.shape[0]
        unique_index_num = unique_emb_index_value.shape[0]

        ######################################
        #   teacher
        ######################################

        teacher_batch, masked_entity_label = self.create_teacher_batch(
            batch=batch, unique_emb_index_value=unique_emb_index_value)

        if self.teacher is not None:
            # teacher score
            teacher_return: GlobalLink_BaseModelReturn = self.teacher(teacher_batch)

            # teacher entity emb

        else:
            teacher_return = None


        ######################################
        #   student
        ######################################

        # # # 得到每个unique_emb_index_value对应的entity description
        # unique_emb_index_value中每个value在batch.teacher_emb_index中第一次出现的位置

        unique_emb_index_first_occur = get_first_occur(
            unique_value=unique_emb_index_value, values=teacher_emb_index)  # [unique_index_num]
        assert (teacher_emb_index[unique_emb_index_first_occur] == unique_emb_index_value).all()

        # 用刚才得到的第一次出现的位置索引出相应的entity description

        unique_entity_text = WordBertInput(*[field[unique_emb_index_first_occur] for field in batch.entity_text])
        # unique_entity_text
        #   fields: input_ids, token_type_ids, attention_mask
        #   fields shape:  [unique_index_num, seq_len]

        student_return: GlobalLink_BaseModelReturn = self.student(
            context_word_seq=batch.luke_context_word_seq,
            context_entity_seq=batch.luke_context_entity_seq,
            candidate_word_seq=unique_entity_text,
            masked_entity_label=masked_entity_label,
            unique_entity_label_value=unique_emb_index_value,
        )

        ######################################
        #   loss
        ######################################

        loss_total = torch.tensor(0., dtype=student_return.loss.dtype, device=self.device)

        # # # link loss
        student_link_loss = student_return.loss
        loss_total += student_link_loss * self.student_link_loss_weight

        # # # score distill loss
        if self.teacher is not None:
            assert (student_return.label == teacher_return.label).all()
            # student_return.predict_score, teacher_return.predict_score: [batch_masked_entity_num, cand_num]
            distill_log_softmax_p = F.log_softmax(student_return.predict_score, dim=1)
            # [batch_masked_entity_num, cand_num]
            teacher_score = torch.stack(teacher_return.predict_score, dim=0) if isinstance(teacher_return.predict_score, List) \
                else teacher_return.predict_score
            distill_softmax_temperature_q = F.softmax(teacher_score / self.distill_temperature, dim=1)
            # [batch_masked_entity_num, cand_num]
            score_distill_loss = self.kl_div_loss(input=distill_log_softmax_p, target=distill_softmax_temperature_q)
            loss_total += score_distill_loss * self.score_distill_loss_weight
        else:
            score_distill_loss = torch.tensor(np.nan, device=self.device)

        # # # emb distill loss
        if self.emb_distill_head is not None:
            emb_distill_ce_loss, emb_distill_mse_loss, emb_distill_invert_ce_loss, pred_classify_top1_score, pred_classify_top1_index = self.emb_distill_head(
                entity_emb=student_return.entity_emb, entity_emb_index=unique_emb_index_value
            )
            loss_total += emb_distill_mse_loss * self.emb_distill_mse_loss_weight
            loss_total += emb_distill_ce_loss * self.emb_distill_ce_loss_weight
            loss_total += emb_distill_invert_ce_loss * self.emb_distill_invert_ce_loss_weight
        else:
            emb_distill_ce_loss, emb_distill_mse_loss, emb_distill_invert_ce_loss = \
                torch.tensor(np.nan, device=self.device), torch.tensor(np.nan, device=self.device), torch.tensor(np.nan, device=self.device)

        ######################################
        #   prepare return
        ######################################

        if batch.wikipedia_pageid is not None:
            wikipedia_pageid = batch.wikipedia_pageid
        else:
            wikipedia_pageid = list(chain.from_iterable(batch.luke_masked_entity_label.gold_wikipedia_pageid))
        wikipedia_pageid = torch.tensor(wikipedia_pageid)
        unique_wikipedia_pageid = wikipedia_pageid[unique_emb_index_first_occur]  # [unique_index_num]

        student_entity_emb = student_return.entity_emb
        student_entity_bias=torch.ones(student_entity_emb.shape[0], dtype=student_entity_emb.dtype, device=self.device)
        model_return = Distill_BaseModelReturn(
            batch_size=batch_size,
            loss_total=loss_total,

            loss_student_link=student_link_loss,
            loss_score_distill=score_distill_loss,
            loss_bias=torch.tensor(np.nan, device=self.device),
            loss_emb_ce=emb_distill_ce_loss,
            loss_emb_mse=emb_distill_mse_loss,
            loss_emb_invert_ce=emb_distill_invert_ce_loss,

            wikipedia_pageid=unique_wikipedia_pageid,
            student_entity_emb=student_entity_emb,
            student_entity_bias=student_entity_bias,

            student_label=student_return.label,
            student_predict_score=student_return.predict_score,
            student_out_topk_predict_score=student_return.out_topk_predict_score,
            student_out_topk_indices=student_return.out_topk_indices,
        )

        return model_return

    def create_teacher_batch(self, batch: Distill_ModelExample, unique_emb_index_value: torch.Tensor):
        batch_size = batch.luke_context_word_seq.input_ids.shape[0]

        if batch.luke_masked_entity_label is not None:
            mention_num = batch.luke_masked_entity_label.mlm_mask.sum()
            masked_entity_label = batch.luke_masked_entity_label._replace(
                candidate_vocab_index=tuple([
                    tuple([
                        unique_emb_index_value.tolist(),  # unique_emb_index_value.tolist(),  # slice(4, None)
                    ])
                    for _ in range(mention_num)
                ]),
            )

        else:
            masked_entity_label = GlobalLink_EntityMLMLabel(
                query_id=tuple(
                    tuple([
                        f"fake:{random.randint(0, 1_000_000_000_000)}"
                    ])
                    for _ in range(batch_size)
                ),
                has_gold=tuple(
                    tuple([True])
                    for _ in range(batch_size)
                ),  # Tuple[Tuple[int]]
                mlm_label=eop.rearrange(batch.teacher_emb_index, "batch -> batch 1"),  # [batch_size, max_entity_num]
                mlm_mask=torch.ones_like(batch.luke_context_entity_seq.input_ids),  # [batch_size, max_entity_num]
                gold_wikipedia_pageid=tuple(
                    tuple([pageid])
                    for pageid in batch.wikipedia_pageid
                ),  # Tuple[Tuple[int]]
                gold_wikidata_id=tuple(
                    tuple([f"Q{pageid}"])
                    for pageid in batch.wikipedia_pageid
                ),  # Tuple[Tuple[str]]
                luke_freq_bin=tuple((-1,) for _ in range(batch_size)),  # Tuple[Tuple[int]]
                candidate_vocab_index=tuple([
                    tuple([
                        unique_emb_index_value.tolist(),  # slice(4, None)
                    ])
                    for _ in range(batch_size)
                ]),
                # Tuple[Tuple[List[int]]], 三层嵌套：batch的每个context -> context中每个mention(这里正好只有一个) -> 每个mention的候选entity
            )

        teacher_batch = GlobalLink_ModelExample(
            doc_id=["fake" for _ in range(batch_size)],
            word_seq=batch.luke_context_word_seq,
            entity_seq=batch.luke_context_entity_seq,
            masked_entity_label=masked_entity_label,
            masked_word_label=None,
        )
        return teacher_batch, masked_entity_label

    def entity_vocab_size(self):
        return self.teacher.entity_vocab_size() if self.teacher is not None else None
