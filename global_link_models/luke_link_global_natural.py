from typing import List, Dict, Tuple, Union

import numpy as np
from typing_extensions import Literal
from pathlib import Path
from logging import Logger
from itertools import chain, groupby

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

try:
    from transformers.modeling_bert import ACT2FN, BertLayerNorm, BertPreTrainingHeads
    from transformers.modeling_roberta import RobertaLMHead
except ModuleNotFoundError:
    from transformers.models.bert.modeling_bert import ACT2FN, BertPreTrainingHeads
    from transformers.models.roberta.modeling_roberta import RobertaLMHead
    from torch.nn import LayerNorm as BertLayerNorm
from transformers import AutoConfig, AutoModelForPreTraining, \
    PreTrainedTokenizerFast, PreTrainedTokenizer

from luke_model.luke_model import LukeModel, LukeConfig
from global_link_models.model_type_hints import BaseModelReturn
from global_link_dataset import ModelExample, WordBertInput, EntityBertInput, EntityMLMLabel, WordMLMLabel
from utils.freeze_model import freeze, un_freeze
from global_link_models.luke_link_local import EntityPredictionHeadTransform, EntityPredictionHead_FullMatrix, EntityPredictionHead_CandidateList


class LukeLinkGlobalNatural(LukeModel):
    def __init__(
            self,
            pretrained_dir: Union[str, Path],
            out_topk_candidates: int,
            entity_vocab_size: int,
            entity_emb_size: int,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            py_logger: Logger,
            freeze_bert: bool,
            freeze_entity: bool,
            freeze_entity_prediction: bool,
            use_entity_bias: bool,
            candidate_from: Literal['list', 'full'],
            *args,
            **kwargs,
    ):

        bert_config = AutoConfig.from_pretrained(pretrained_dir)
        bert_model_name = pretrained_dir.name if isinstance(pretrained_dir, Path) else pretrained_dir.split("/")[-1]
        config = LukeConfig(
            entity_vocab_size=entity_vocab_size,
            bert_model_name=bert_model_name,
            entity_emb_size=entity_emb_size,
            **bert_config.to_dict(),
        )
        super().__init__(config)

        self.py_logger = py_logger
        self.tokenizer = tokenizer
        self.out_topk_candidates = out_topk_candidates
        # self.allow_nil = allow_nil

        self.special_entity_index = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[MASK]": 2,  # 实际用的mask
            "[MASK2]": 3,  # 似乎没用到？
        }

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.lm_head = RobertaLMHead(config)  # 有bias，bias与Embedding层不共享，weight中不包括bias
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight
        else:
            self.cls = BertPreTrainingHeads(config)
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.candidate_from = candidate_from
        if candidate_from == 'full':
            self.entity_predictions = EntityPredictionHead_FullMatrix(
                config=config, use_entity_bias=use_entity_bias)  # 有bias，bias与Embedding层不共享，weight中不包括bias
        elif candidate_from == 'list':
            self.entity_predictions = EntityPredictionHead_CandidateList(
                config=config, use_entity_bias=use_entity_bias)  # 有bias，bias与Embedding层不共享，weight中不包括bias
        else:
            raise ValueError(candidate_from)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.train_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.eval_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.apply(self.init_weights)

        # init: load bert weight
        bert_model = AutoModelForPreTraining.from_pretrained(pretrained_dir)
        bert_state_dict = bert_model.state_dict()
        self.load_bert_weights(state_dict=bert_state_dict)

        freeze(self.pooler)
        freeze(self.lm_head)
        if self.embeddings.word_embeddings.weight is self.lm_head.decoder.weight:
            un_freeze(self.embeddings.word_embeddings.weight)
        if freeze_bert:
            freeze(self.embeddings)
            freeze(self.encoder)
        if freeze_entity or freeze_entity_prediction:
            freeze(self.entity_embeddings.entity_embeddings)
            if isinstance(self.entity_predictions.bias, nn.Parameter):
                freeze(self.entity_predictions.bias)
        if freeze_entity_prediction:  # 前面已经freeze了entity
            assert freeze_entity
            freeze(self.entity_predictions)
        pass

    def forward(self, batch: ModelExample):

        entity_ids = batch.entity_seq.input_ids.clone()

        # masked_entity_label.mlm_label: [batch_size, max_entity_num]
        entity_mask = batch.masked_entity_label.mlm_label != -100
        # entity_mask: [batch_size, max_entity_num]

        assert (entity_mask == batch.masked_entity_label.mlm_mask).all()

        array_entity_mask: np.ndarray = entity_mask.cpu().numpy()  # array[bool]
        array_predict_score: np.ndarray = array_entity_mask.copy().astype(int).astype(object)  # array[object]
        array_out_topk_predict_score: np.ndarray = array_entity_mask.copy().astype(int).astype(object)  # array[object]
        array_out_topk_indices: np.ndarray = array_entity_mask.copy().astype(int).astype(object)  # array[object]

        while entity_mask.sum() > 0:

            seq_index_2_coordinate: List[Tuple[int, int]] = [tuple(co) for co in torch.nonzero(input=entity_mask).tolist()]  # entity mask中每个非0元素的坐标
            # seq_index_2_coordinate: [2, current_mask_entity_num]

            coordinate_2_seq_index = {co: i for i, co in enumerate(seq_index_2_coordinate)}

            ###########################################
            #   encode sequence
            ###########################################

            word_sequence_output, entity_sequence_output = self.encode_word_entity_sequence(
                word_ids=batch.word_seq.input_ids,
                word_segment_ids=batch.word_seq.token_type_ids,
                word_attention_mask=batch.word_seq.attention_mask,

                entity_ids=entity_ids,
                entity_position_ids=batch.entity_seq.position_ids,
                entity_segment_ids=batch.entity_seq.token_type_ids,
                entity_attention_mask=batch.entity_seq.attention_mask,
            )
            # word_sequence_output: [batch_size, context_seq_len, hidden_size]
            # entity_sequence_output: [batch_size, max_entity_num, hidden_size]

            ###########################################
            #   select current predict spans
            ###########################################

            target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
            # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
            # target_entity_sequence_output: [current_mask_entity_num * hidden_size]

            target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
            # [current_mask_entity_num, hidden_size]

            # batch.masked_entity_label.candidate_vocab_index:
            #   Tuple[Tuple[slice]] or Tuple[Tuple[List[int]]], len: batch_size * mention_num * cand_num
            list_candidate_emb_index = [batch.masked_entity_label.candidate_vocab_index[i][j] for i,j in seq_index_2_coordinate]
            # List[slice] or List[List[int]], len: current_mask_entity_num * cand_num

            assert len(list_candidate_emb_index) == len(target_entity_sequence_output)

            ###########################################
            #   score
            ###########################################

            if self.candidate_from == 'full':
                entity_scores, topk_scores, topk_indices = self.score_from_full_matrix(
                    list_candidate_emb_index=list_candidate_emb_index,
                    target_entity_sequence_output=target_entity_sequence_output,
                )
                # entity_scores: [current_mask_entity_num, entity_vocab_size]
                # topk_scores, topk_indices: [current_mask_entity_num, out_top_k]
            elif self.candidate_from == 'list':
                entity_scores, topk_scores, topk_indices = self.score_from_candidate_list(
                    list_candidate_emb_index=list_candidate_emb_index,
                    target_entity_sequence_output=target_entity_sequence_output,
                )
            else:
                raise ValueError(self.candidate_from)

            ###########################################
            # select current iter prediction
            ###########################################

            list_current_predict_coordinate = []
            for key, group in groupby(seq_index_2_coordinate, key=lambda co: co[0]):
                list_current_predict_coordinate.append(list(group)[0])

            for coo in list_current_predict_coordinate:
                seq_index = coordinate_2_seq_index[coo]
                array_predict_score[coo[0], coo[1]] = entity_scores[seq_index]
                array_out_topk_predict_score[coo[0], coo[1]] = topk_scores[seq_index]
                array_out_topk_indices[coo[0], coo[1]] = topk_indices[seq_index]

            pass  # 用来打断点的

            ###########################################
            # prepare next iter data
            ###########################################

            for coo in list_current_predict_coordinate:
                seq_index = coordinate_2_seq_index[coo]
                entity_ids[coo[0], coo[1]] = topk_indices[seq_index][0]
                entity_mask[coo[0], coo[1]] = False

            pass  # 用来打断点的

        ###########################################
        #   prepare return
        ###########################################

        target_entity_labels = torch.masked_select(batch.masked_entity_label.mlm_label, batch.masked_entity_label.mlm_mask.bool())
        # [batch_mask_entity_num]

        predict_score = array_predict_score[array_entity_mask.astype(bool)].tolist()
        out_topk_predict_score = array_out_topk_predict_score[array_entity_mask.astype(bool)].tolist()
        out_topk_indices = array_out_topk_indices[array_entity_mask.astype(bool)].tolist()

        model_return = BaseModelReturn(
            query_id=list(chain.from_iterable(batch.masked_entity_label.query_id)),
            has_gold=list(chain.from_iterable(batch.masked_entity_label.has_gold)),
            freq_bin=list(chain.from_iterable(batch.masked_entity_label.luke_freq_bin)),
            gold_wikidata_id=list(chain.from_iterable(batch.masked_entity_label.gold_wikidata_id)),
            gold_wikipedia_pageid=list(chain.from_iterable(batch.masked_entity_label.gold_wikipedia_pageid)),
            label=target_entity_labels,

            predict_score=predict_score,
            out_topk_predict_score=out_topk_predict_score,
            out_topk_indices=out_topk_indices,

            loss=torch.tensor(np.nan),  # todo
        )

        return model_return

    def encode_word_entity_sequence(
            self,
            word_ids: torch.Tensor,
            word_segment_ids: torch.Tensor,
            word_attention_mask: torch.Tensor,

            entity_ids: torch.Tensor,
            entity_position_ids: torch.Tensor,
            entity_segment_ids: torch.Tensor,
            entity_attention_mask: torch.Tensor,
    ):
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
        output = super().forward(
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,

            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

        # encode context sequence and additional (masked) entity sequence
        word_sequence_output, entity_sequence_output = output[:2]
        # word_sequence_output: [batch_size, context_seq_len, hidden_size]
        # entity_sequence_output: [batch_size, max_entity_num, hidden_size]

        return word_sequence_output, entity_sequence_output


    def score_from_full_matrix(
            self,
            list_candidate_emb_index: List[Union[slice, List[int]]],
            target_entity_sequence_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ###########################################
        #   pred score for each mention
        ###########################################

        # predict on full entity emb matrix
        all_pred_entity_scores = self.entity_predictions(target_entity_sequence_output)  # [batch_mask_entity_num, entity_vocab_size]
        all_pred_entity_scores = all_pred_entity_scores.view(-1, self.config.entity_vocab_size)  # [batch_mask_entity_num, entity_vocab_size]

        # select candidate score only
        entity_scores = torch.ones_like(all_pred_entity_scores) * -10_000.  # [batch_mask_entity_num, entity_vocab_size]

        assert len(list_candidate_emb_index) == entity_scores.shape[0]
        for i, c in enumerate(list_candidate_emb_index):
            c: Union[slice, List]
            entity_scores[i][c] = all_pred_entity_scores[i][c]

        ###########################################
        #   get topk
        ###########################################

        topk_scores, topk_indices = entity_scores.topk(k=self.out_topk_candidates, dim=1)
        # topk_scores, topk_indices: [batch_mask_entity_num, out_top_k]

        assert topk_scores.shape[0] == target_entity_sequence_output.shape[0] == len(list_candidate_emb_index)

        return entity_scores, topk_scores, topk_indices

    def score_from_candidate_list(
            self,
            list_candidate_emb_index: List[Union[slice, List[int]]],
            target_entity_sequence_output: torch.Tensor,
    ):
        raise NotImplementedError()

    def entity_vocab_size(self):
        return self.entity_embeddings.entity_embeddings.weight.shape[0]
