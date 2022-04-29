from typing import List, Union
from pathlib import Path
from logging import Logger
from itertools import chain

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

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
from global_link_models.luke_link_local import EntityPredictionHead_FullMatrix as EntityPredictionHead
from global_link_models.model_type_hints import BaseModelReturn
from global_link_dataset import ModelExample
from global_link_models.listwise_ranking_head import TransformerListwisePredictHead, IdentityPredictHead
from utils.freeze_model import freeze, un_freeze


class LukeLinkLocalListwise(LukeModel):
    def __init__(
            self,
            pretrained_dir: Union[str, Path],
            in_topk_candidates: int,
            out_topk_candidates: int,
            max_context_len: int,
            max_entity_len: int,
            max_span_len: int,
            mention_add_start_end: bool,
            model_data_format: str,
            entity_vocab_table_name: str,
            entity_vocab_size: int,
            entity_emb_size: int,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            allow_nil: bool,
            seed: int,
            py_logger: Logger,
            freeze_entity: bool,
            freeze_bert: bool,
            freeze_pointwise_entity_prediction: bool,
            listwise_num_self_attention_layers: int,
            listwise_score_mode: str,
            listwise_score_fc_norm: bool,
            listwise_score_fc_xavier: bool,
            listwise_use_position_emb: bool,
            listwise_use_score_emb: bool,
            listwise_use_query_hidden_size: bool,
            fineturn_pointwise: bool,
            fineturn_listwise: bool,
            output_listwise_rerank: bool,
            use_entity_bias: bool,
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
        self.allow_nil = allow_nil

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

        self.entity_predictions = EntityPredictionHead(config=config, use_entity_bias=use_entity_bias)  # 有bias，bias与Embedding层不共享，weight中不包括bias
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.listwise_entity_predictions = TransformerListwisePredictHead(
            config=bert_config,
            entity_emb_size=config.entity_emb_size,
            max_position_emb=out_topk_candidates,
            num_self_attention_layers=listwise_num_self_attention_layers,
            score_mode=listwise_score_mode,
            score_fc_norm=listwise_score_fc_norm,
            score_fc_xavier=listwise_score_fc_xavier,
            use_position_emb=listwise_use_position_emb,
            use_score_emb=listwise_use_score_emb,
            use_query_hidden_size=listwise_use_query_hidden_size,
        )

        self.train_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.eval_loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.apply(self.init_weights)

        # init: load bert weight
        bert_model = AutoModelForPreTraining.from_pretrained(pretrained_dir)
        bert_state_dict = bert_model.state_dict()
        self.load_bert_weights(state_dict=bert_state_dict)

        # freeze parameter
        freeze(self.pooler)
        freeze(self.lm_head)
        if self.embeddings.word_embeddings.weight is self.lm_head.decoder.weight:
            un_freeze(self.embeddings.word_embeddings.weight)
        if freeze_bert:
            freeze(self.embeddings)
            freeze(self.encoder)
        if freeze_entity or freeze_pointwise_entity_prediction:
            freeze(self.entity_embeddings.entity_embeddings)
            if isinstance(self.entity_predictions.bias, nn.Parameter):
                freeze(self.entity_predictions.bias)
        if freeze_pointwise_entity_prediction:
            assert freeze_entity
            freeze(self.entity_predictions)

        assert freeze_pointwise_entity_prediction ^ fineturn_pointwise, \
            (freeze_pointwise_entity_prediction, fineturn_pointwise)

        if not fineturn_listwise:
            freeze(self.listwise_entity_predictions)

        self.fineturn_pointwise: bool = fineturn_pointwise
        self.fineturn_listwise: bool = fineturn_listwise
        self.output_listwise_rerank: bool = output_listwise_rerank

        # freeze_sub_model_list: List[nn.Module] = []
        # if freeze_bert:
        #     freeze_sub_model_list += [
        #         self.embeddings,  # 只是bert三个Embedding，不包括Entity Embedding
        #         self.encoder,
        #         self.pooler,
        #         self.lm_head,
        #     ]
        # if freeze_entity:
        #     freeze_sub_model_list += [
        #         self.entity_embeddings,
        #         self.entity_predictions,
        #     ]
        # for sub_model in freeze_sub_model_list:
        #     for param in sub_model.parameters():
        #         param.requires_grad = False

    def entity_vocab_size(self):
        return self.entity_embeddings.entity_embeddings.weight.shape[0]

    def forward(self, batch: ModelExample):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility
        ignore_entity_score = -10_000.  # mask score for entity not in given candidate list

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
            word_ids=batch.word_seq.input_ids,
            word_segment_ids=batch.word_seq.token_type_ids,
            word_attention_mask=batch.word_seq.attention_mask,

            entity_ids=batch.entity_seq.input_ids,
            entity_position_ids=batch.entity_seq.position_ids,
            entity_segment_ids=batch.entity_seq.token_type_ids,
            entity_attention_mask=batch.entity_seq.attention_mask,
        )

        # encode context sequence and additional (masked) entity sequence
        word_sequence_output, entity_sequence_output = output[:2]
        # word_sequence_output: [batch_size, context_seq_len, hidden_size]
        # entity_sequence_output: [batch_size, max_entity_num, hidden_size]

        # get masked entity score
        # masked_entity_label.mlm_label: [batch_size, max_entity_num]
        entity_mask = batch.masked_entity_label.mlm_label != -100
        # entity_mask: [batch_size, max_entity_num]
        assert (entity_mask == batch.masked_entity_label.mlm_mask).all()

        target_query_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
        # entity_mask.unsqueeze(-1): [batch_size, max_entity_num, 1]
        # target_query_sequence_output: [batch_mask_entity_num * hidden_size]

        target_query_sequence_output = target_query_sequence_output.view(-1, self.config.hidden_size)
        # [batch_mask_entity_num, hidden_size]

        # luke pointwise pred score
        all_pred_entity_scores = self.entity_predictions(target_query_sequence_output)  # [batch_mask_entity_num, entity_vocab_size]
        all_pred_entity_scores = all_pred_entity_scores.view(-1, self.config.entity_vocab_size)  # [batch_mask_entity_num, entity_vocab_size]

        # select candidate score only
        pointwise_scores = torch.ones_like(all_pred_entity_scores) * ignore_entity_score  # [batch_mask_entity_num, entity_vocab_size]
        all_candidate_vocab_index = list(chain.from_iterable(batch.masked_entity_label.candidate_vocab_index))
        assert len(all_candidate_vocab_index) == pointwise_scores.shape[0]
        for i, c in enumerate(all_candidate_vocab_index):
            c: Union[slice, List]
            pointwise_scores[i][c] = all_pred_entity_scores[i][c]

        # select top-k candidate
        topk_pointwise_scores, topk_pointwise_indices = \
            pointwise_scores.topk(k=self.out_topk_candidates, dim=1)
        topk_candidate_attention_mask = (topk_pointwise_scores > ignore_entity_score).long()
        # topk_pointwise_scores, topk_pointwise_indices, topk_candidate_attention_mask:
        # [batch_mask_entity_num, out_top_k]

        pointwise_score_labels = torch.masked_select(batch.masked_entity_label.mlm_label, entity_mask)
        # [batch_mask_entity_num]

        # replace last candidate entity with gold entity, if there's no gold entity
        if self.training:
            pointwise_score_one_hot = (
                    topk_pointwise_indices ==
                    pointwise_score_labels.unsqueeze(-1).expand_as(topk_pointwise_indices)
            ).long()  # [batch_mask_entity_num, out_top_k]
            pointwise_no_gold = (pointwise_score_one_hot.sum(dim=1) == 0).bool()  # [batch_mask_entity_num]
            replace_last_candidate = pointwise_no_gold & (pointwise_score_labels != -1)  # 如果不允许有nil实体，则nil的label是-1，在计算损失时会略过。但是pointwise_scores.gather的index不允许有负值否则cuda error
            topk_pointwise_indices[replace_last_candidate, -1] = pointwise_score_labels[replace_last_candidate]  # todo
            topk_pointwise_scores = pointwise_scores.gather(dim=1, index=topk_pointwise_indices)  # re-select score

        topk_candidate_embeddings = self.entity_embeddings.entity_embeddings(topk_pointwise_indices)
        # [batch_mask_entity_num, out_top_k, entity_emb_dim]

        # listwise predict
        listwise_scores = self.listwise_entity_predictions(
            query=target_query_sequence_output,
            candidate_entity=topk_candidate_embeddings,
            candidate_entity_score=topk_pointwise_scores,
            candidate_attention_mask=topk_candidate_attention_mask,
        )  # [batch_mask_entity_num, out_top_k]

        # compute loss

        # pointwise score loss
        pointwise_score_loss_fn = self.train_loss if self.training else self.eval_loss
        pointwise_score_loss = pointwise_score_loss_fn(input=pointwise_scores, target=pointwise_score_labels)

        # listwise score loss
        listwise_score_one_hot = (
                topk_pointwise_indices ==
                pointwise_score_labels.unsqueeze(-1).expand_as(topk_pointwise_indices)
        ).long()  # [batch_mask_entity_num, out_top_k]
        if self.training:
            if self.allow_nil:
                assert (listwise_score_one_hot.sum(dim=1) == 1).all()
            else:
                assert (listwise_score_one_hot.sum(dim=1) <= 1).all()
        else:
            assert (listwise_score_one_hot.sum(dim=1) <= 1).all()
        listwise_no_gold = (listwise_score_one_hot.sum(dim=1) == 0)
        listwise_score_labels = listwise_score_one_hot.argmax(dim=1)  # [batch_mask_entity_num]
        listwise_score_labels[listwise_no_gold] = -1
        listwise_score_loss_fn = self.train_loss if self.training else self.eval_loss
        listwise_score_loss = listwise_score_loss_fn(input=listwise_scores, target=listwise_score_labels)

        loss = torch.tensor(0., dtype=listwise_score_loss.dtype, device=listwise_score_loss.device)
        if self.fineturn_listwise:
            loss += listwise_score_loss
        if self.fineturn_pointwise:
            loss += pointwise_score_loss


        # rerank listwise score again
        topk_listwise_scores, topk_listwise_indices = listwise_scores.topk(k=self.out_topk_candidates, dim=1)
        assert (listwise_scores.gather(dim=1, index=topk_listwise_indices) == topk_listwise_scores).all()

        if self.output_listwise_rerank:
            topk_scores = topk_listwise_scores
            topk_indices = topk_pointwise_indices.gather(dim=1, index=topk_listwise_indices)
        else:
            topk_scores = topk_pointwise_scores
            topk_indices = topk_pointwise_indices

        model_return = BaseModelReturn(
            query_id=list(chain.from_iterable(batch.masked_entity_label.query_id)),
            has_gold=list(chain.from_iterable(batch.masked_entity_label.has_gold)),
            freq_bin=list(chain.from_iterable(batch.masked_entity_label.luke_freq_bin)),
            gold_wikidata_id=list(chain.from_iterable(batch.masked_entity_label.gold_wikidata_id)),
            gold_wikipedia_pageid=list(chain.from_iterable(batch.masked_entity_label.gold_wikipedia_pageid)),
            label=pointwise_score_labels.detach().cpu().numpy(),
            predict_score=None,
            out_topk_predict_score=topk_scores.detach().cpu().numpy(),
            out_topk_indices=topk_indices.detach().cpu().numpy(),
            loss=loss,
        )

        return model_return

