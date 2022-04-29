from typing import List, Union
from typing_extensions import Literal
from pathlib import Path
from logging import Logger
from itertools import chain

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


class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityPredictionHead_FullMatrix(nn.Module):
    def __init__(self, config: LukeConfig, use_entity_bias: bool):
        super(EntityPredictionHead_FullMatrix, self).__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)  # 全连接，激活，LayerNorm
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size)) if use_entity_bias else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)  # [batch_size, hidden_size]
        full_pred_score = self.decoder(hidden_states) + self.bias  # [batch_size, full_entity_num]
        return full_pred_score


class EntityPredictionHead_CandidateList(nn.Module):
    def __init__(self, config: LukeConfig, use_entity_bias: bool):
        super(EntityPredictionHead_CandidateList, self).__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)  # 全连接，激活，LayerNorm
        self.decoder = nn.Embedding(num_embeddings=config.entity_vocab_size, embedding_dim=config.entity_emb_size)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size)) if use_entity_bias else 0

    def forward(self, hidden_states: torch.Tensor, list_cand_emb_index: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        assert hidden_states.shape[0] == len(list_cand_emb_index)

        hidden_states = self.transform(hidden_states)  # [batch_size, hidden_size]
        # pred_score = self.decoder(hidden_states) + self.bias

        # for循环版本
        list_pred_score = []
        for cand_emb_index, hid_st in zip(list_cand_emb_index, hidden_states):
            cand_emb = self.decoder(cand_emb_index)  # [cand_num, hidden_size]
            bias = self.bias[cand_emb_index] if isinstance(self.bias, torch.Tensor) else self.bias  # [cand_num]
            pred_score = torch.einsum("c h, h -> c", cand_emb, hid_st) + bias  # [cand_num]
            list_pred_score.append(pred_score)

        # # 矩阵乘版本
        # list_cand_num = [len(cand_emb_index) for cand_emb_index in list_cand_emb_index]  # len: batch_size
        # sub_embeddings = torch.zeros(
        #     size=(batch_size, max(list_cand_num), self.config.entity_emb_size),
        #     device=self.decoder.weight.device, dtype=self.decoder.weight.dtype
        # )  # [batch_size, max_cand_num, hidden_size]
        # sub_bias = torch.zeros(
        #     size=(batch_size, max(list_cand_num)),
        #     device=self.decoder.weight.device, dtype=self.decoder.weight.dtype
        # )  # [batch_size, max_cand_num]
        # for i, cand_emb_index in enumerate(list_cand_emb_index):
        #     sub_embeddings[i][:len(cand_emb_index)] = self.decoder(cand_emb_index)  # [cand_num, hidden_size]
        #     sub_bias[i][:len(cand_emb_index)] = self.bias[cand_emb_index]  # [cand_num]
        # pad_pred_score = torch.einsum("b c h, b h -> b c", sub_embeddings, hidden_states)  # [batch_size, max_cand_num]
        # pad_pred_score = pad_pred_score + sub_bias  # [batch_size, max_cand_num]
        # list_pred_score = [pred_score[:cand_num] for pred_score, cand_num in zip(pad_pred_score, list_cand_num)]

        return list_pred_score


class LukeLinkLocal(LukeModel):
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
        super(LukeLinkLocal, self).__init__(config)

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

        ###########################################
        #   encode sequence
        ###########################################

        word_sequence_output, entity_sequence_output = self.encode_word_entity_sequence(
            word_seq=batch.word_seq, entity_seq=batch.entity_seq
        )
        # word_sequence_output: [batch_size, context_seq_len, hidden_size]
        # entity_sequence_output: [batch_size, max_entity_num, hidden_size]

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
                entity_sequence_output=entity_sequence_output,
            )
        elif self.candidate_from == 'list':
            model_return = self.score_from_candidate_list(
                masked_entity_label=batch.masked_entity_label,
                entity_sequence_output=entity_sequence_output,
            )
        else:
            raise ValueError(self.candidate_from)

        return model_return

    def encode_word_entity_sequence(self, word_seq: WordBertInput, entity_seq: EntityBertInput):
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
        output = super(LukeLinkLocal, self).forward(
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

        return word_sequence_output, entity_sequence_output

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

        target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
        # [batch_mask_entity_num, hidden_size]

        ###########################################
        #   pred score for each mention
        ###########################################

        # predict on full entity emb matrix
        all_pred_entity_scores = self.entity_predictions(target_entity_sequence_output)  # [batch_mask_entity_num, entity_vocab_size]
        all_pred_entity_scores = all_pred_entity_scores.view(-1, self.config.entity_vocab_size)  # [batch_mask_entity_num, entity_vocab_size]

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
        return self.entity_embeddings.entity_embeddings.weight.shape[0]
