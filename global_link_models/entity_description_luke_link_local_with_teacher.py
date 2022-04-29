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
import pandas as pd

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
from utils.utils import current_time_string, init_logger, safely_connect_sqlite


from global_link_models.entity_description_luke_link_local import EntityDescriptionLukeLinkLocal
from global_link_models.luke_link_local import LukeLinkLocal


def extract_state_dict(checkpoint_path: Union[Path, str]):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {re.sub(r"^model\.", "", k): v for k,v in state_dict.items()}
    return state_dict


class EntityDescriptionLukeLinkLocalWithTeacher(pl.LightningModule):
    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            py_logger: Logger,
            entity_db_path: Union[Path, str],

            candidate_from: Literal["full", "list"],
            out_topk_candidates: int,
            distill_temperature: float,
            student_link_loss_weight: float,
            score_distill_loss_weight: float,

            # teacher specific
            teacher_init_checkpoint: Union[Path, str],
            teacher_entity_vocab_table_name: str,
            teacher_entity_vocab_size: int,
            teacher_pretrained_dir: Union[str, Path],
            teacher_entity_emb_size: int,
            teacher_freeze_bert: bool,
            teacher_freeze_entity: bool,
            teacher_freeze_entity_prediction: bool,
            teacher_use_entity_bias: bool,

            # student specific
            student_init_checkpoint: Union[Path, str],
            student_entity_encoding_path: Union[Path, str],
            student_entity_vocab_table_name: str,
            student_entity_vocab_size: int,
            student_candidate_encoding_device: Optional[str],
            student_bert_pretrained_dir: Union[Path, str],
            student_luke_pretrained_dir: Union[Path, str],
            student_share_backbone: bool,
            student_entity_encoder_use_luke: bool,
            student_special_entity_emb_size: int,
            student_entity_pooling: Literal["average", "cls", "luke_span"],
            student_normalize_entity_encoding: bool,
            student_do_transform: bool,

            *args,
            **kwargs,
    ):
        super().__init__()

        self.distill_temperature = distill_temperature

        # loss
        assert 0 <= student_link_loss_weight <= 1, student_link_loss_weight
        assert 0 <= score_distill_loss_weight <= 1, score_distill_loss_weight
        assert student_link_loss_weight + score_distill_loss_weight == 1, \
            (student_link_loss_weight, score_distill_loss_weight)

        self.student_link_loss_weight = student_link_loss_weight
        self.score_distill_loss_weight = score_distill_loss_weight

        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        # teacher
        self.teacher: Optional[nn.Module] = None
        self.student_label_2_teacher_label: Dict[int, int] = None
        if teacher_init_checkpoint:
            teacher = LukeLinkLocal(
                tokenizer=tokenizer,
                py_logger=py_logger,

                out_topk_candidates=out_topk_candidates,
                candidate_from=candidate_from,

                pretrained_dir=teacher_pretrained_dir,
                entity_vocab_size=teacher_entity_vocab_size,
                entity_emb_size=teacher_entity_emb_size,
                freeze_bert=teacher_freeze_bert,
                freeze_entity=teacher_freeze_entity,
                freeze_entity_prediction=teacher_freeze_entity_prediction,
                use_entity_bias=teacher_use_entity_bias,
            )
            teacher_state_dict = extract_state_dict(checkpoint_path=teacher_init_checkpoint)
            teacher.load_state_dict(teacher_state_dict)
            self.teacher = teacher
            freeze(self.teacher)

            entity_db_con = safely_connect_sqlite(db_path=entity_db_path)
            sql = f"""
                select
                    student_entity_vocab.wikipedia_pageid as wikipedia_pageid,
                    student_entity_vocab.emb_index as student_emb_index,
                    teacher_entity_vocab.emb_index as teacher_emb_index
                from
                    {student_entity_vocab_table_name} as student_entity_vocab
                    inner join
                    {teacher_entity_vocab_table_name} as teacher_entity_vocab
                    on student_entity_vocab.wikipedia_pageid=teacher_entity_vocab.wikipedia_pageid
                where
                    student_entity_vocab.wikipedia_pageid>0
            """
            df = pd.read_sql(sql=sql, con=entity_db_con)
            df = pd.concat([
                pd.DataFrame([{"student_emb_index": i, "teacher_emb_index": i} for i in range(4)]),
                df[["student_emb_index", "teacher_emb_index"]]
            ], ignore_index=True).sort_values(by="student_emb_index")
            self.student_label_2_teacher_label = df.set_index("student_emb_index")["teacher_emb_index"].to_dict()

        # student
        student = EntityDescriptionLukeLinkLocal(
            py_logger=py_logger,
            tokenizer=tokenizer,

            out_topk_candidates=out_topk_candidates,
            candidate_from=candidate_from,

            entity_encoding_path=student_entity_encoding_path,
            candidate_encoding_device=student_candidate_encoding_device,
            bert_pretrained_dir=student_bert_pretrained_dir,
            luke_pretrained_dir=student_luke_pretrained_dir,
            share_backbone=student_share_backbone,
            entity_encoder_use_luke=student_entity_encoder_use_luke,
            special_entity_emb_size=student_special_entity_emb_size,
            entity_pooling=student_entity_pooling,
            normalize_entity_encoding=student_normalize_entity_encoding,
            do_transform=student_do_transform,
        )
        student_state_dict = extract_state_dict(checkpoint_path=student_init_checkpoint)
        student.load_state_dict(student_state_dict)
        self.student = student

        if self.teacher is not None:
            assert len(self.student_label_2_teacher_label) == self.student.entity_vocab_size()

        pass

    def forward(self, batch: ModelExample):

        ######################################
        #   student
        ######################################

        student_return: BaseModelReturn = self.student(batch)

        ######################################
        #   loss
        ######################################

        loss_total = torch.tensor(0., dtype=student_return.loss.dtype, device=self.device)

        # # # link loss
        student_link_loss = student_return.loss
        loss_total += student_link_loss * self.student_link_loss_weight

        # # # score distill loss
        if self.teacher is not None:
            teacher_batch = ModelExample(
                doc_id=batch.doc_id,
                word_seq=batch.word_seq,
                masked_word_label=batch.masked_word_label,
                entity_seq=batch.entity_seq._replace(
                    input_ids=batch.entity_seq.input_ids.cpu().apply_(
                        self.student_label_2_teacher_label.get
                    ).to(batch.entity_seq.input_ids.device)
                ),
                masked_entity_label=batch.masked_entity_label._replace(
                    mlm_label=batch.masked_entity_label.mlm_label.cpu().apply_(
                        lambda x: self.student_label_2_teacher_label.get(x, -100)
                    ).to(batch.masked_entity_label.mlm_label.device),
                    candidate_vocab_index=\
                        batch.masked_entity_label.candidate_vocab_index
                        if isinstance(batch.masked_entity_label.candidate_vocab_index[0][0], slice)
                        else [
                            [self.student_label_2_teacher_label.get(x) for x in candidate_vocab_index]
                            for candidate_vocab_index in batch.masked_entity_label.candidate_vocab_index
                        ]
                ),
            )
            teacher_return: BaseModelReturn = self.teacher(teacher_batch)

            assert (student_return.label.cpu().apply_(self.student_label_2_teacher_label.get) == teacher_return.label.cpu()).all()
            # student_return.predict_score, teacher_return.predict_score: [batch_masked_entity_num, cand_num]

            assert isinstance(student_return.predict_score, torch.Tensor) and student_return.predict_score.shape[1] == self.student.entity_vocab_size()
            assert isinstance(teacher_return.predict_score, torch.Tensor) and teacher_return.predict_score.shape[1] == self.teacher.entity_vocab_size()

            pred_score_student = student_return.predict_score[:, list(self.student_label_2_teacher_label.keys())]  # [batch_masked_entity_num, cand_num]
            pred_score_teacher = teacher_return.predict_score[:, list(self.student_label_2_teacher_label.values())]  # [batch_masked_entity_num, cand_num]

            distill_log_softmax_p = F.log_softmax(pred_score_student, dim=1)
            # [batch_masked_entity_num, cand_num]
            distill_softmax_temperature_q = F.softmax(pred_score_teacher / self.distill_temperature, dim=1)
            # [batch_masked_entity_num, cand_num]
            score_distill_loss = self.kl_div_loss(input=distill_log_softmax_p, target=distill_softmax_temperature_q)
            loss_total += score_distill_loss * self.score_distill_loss_weight
        else:
            score_distill_loss = torch.tensor(np.nan, device=self.device)

        model_return = BaseModelReturn(
            query_id=student_return.query_id,
            has_gold=student_return.has_gold,
            freq_bin=student_return.freq_bin,
            gold_wikipedia_pageid=student_return.gold_wikipedia_pageid,
            gold_wikidata_id=student_return.gold_wikidata_id,
            label=student_return.label,
            predict_score=student_return.predict_score,
            out_topk_predict_score=student_return.out_topk_predict_score,
            out_topk_indices=student_return.out_topk_indices,

            loss=loss_total,
            loss_score_distill=score_distill_loss,
            loss_link=student_return.loss,
        )
        return model_return

    def entity_vocab_size(self):
        return self.student.entity_vocab_size()
