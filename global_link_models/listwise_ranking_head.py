from typing import List, Tuple, Dict, NamedTuple, Union, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertPooler
from torch.nn import LayerNorm as BertLayerNorm
import einops as eop


class ListwiseScoreHead(nn.Module):
    def __init__(self, mode: str, fc_xavier: bool, fc_norm: bool, feature_dim: int, dropout: float):
        super(self.__class__, self).__init__()
        assert mode in ("dot", "qd-fc", "d-fc", "bi-linear", "diag-bi-linear"), mode
        self.mode = mode

        self.classifier: nn.Module = None
        self.bi_linear: nn.Module = None
        self.diag_bi_linear: nn.Module = None

        if mode in ("qd-fc", "d-fc"):
            classifier_dim = 2 * feature_dim if mode == "qd-fc" else feature_dim
            if fc_norm:
                self.classifier = nn.Sequential(OrderedDict([
                    ("dropout_0", nn.Dropout(dropout)),
                    ("linear_0", nn.Linear(classifier_dim, classifier_dim)),
                    ("norm_0", nn.LayerNorm(classifier_dim, elementwise_affine=False)),
                    ("relu_0", nn.ReLU()),

                    ("dropout_1", nn.Dropout(dropout)),
                    ("linear_1", nn.Linear(classifier_dim, 1)),
                ]))
            else:
                self.classifier = nn.Sequential(OrderedDict([
                    ("dropout_0", nn.Dropout(dropout)),
                    ("linear_0", nn.Linear(classifier_dim, classifier_dim)),
                    ("relu_0", nn.ReLU()),

                    ("dropout_1", nn.Dropout(dropout)),
                    ("linear_1", nn.Linear(classifier_dim, 1)),
                ]))
            if fc_xavier:
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(self.classifier.linear_0.weight, gain=gain)
                nn.init.xavier_normal_(self.classifier.linear_1.weight, gain=gain)

        elif mode == "bi-linear":  # 双线性型，dot的变体
            # self.bi_linear = nn.Bilinear(in1_features=feature_dim, in2_features=feature_dim, out_features=1, bias=False)
            self.bi_linear = nn.Parameter(torch.eye(feature_dim, dtype=torch.float), requires_grad=True)
        elif mode == "diag-bi-linear":  # 只有对角线有权重的双线性型，相当于加权的dot
            self.diag_bi_linear = nn.Parameter(torch.ones(feature_dim, dtype=torch.float), requires_grad=True)
        else:
            pass

    def forward(
            self,
            query: torch.Tensor,
            candidate_entity: torch.Tensor,
            candidate_attention_mask: torch.Tensor,
    ):
        # query: [batch_size, hidden_size]
        # candidate_entity: [batch_size, candidate_num, hidden_size]
        # candidate_attention_mask: [batch_size, candidate_num]， 注：我用的还是1/0 mask, 不是0./-10000. mask

        expand_query = query.unsqueeze(1).expand_as(candidate_entity)  # [batch_size, candidate_num, hidden_size]
        if self.mode == "dot":
            score = torch.einsum("bch,bch->bc", expand_query, candidate_entity)
        elif self.mode == "qd-fc":
            feature = torch.cat([expand_query, candidate_entity], dim=2)  # [batch_size, candidate_num, hidden_size*2]
            score = self.classifier(feature).squeeze(-1)  # [batch_size, candidate_num]
        elif self.mode == "d-fc":
            feature = candidate_entity  # [batch_size, candidate_num, hidden_size]
            score = self.classifier(feature).squeeze(-1)  # [batch_size, candidate_num]
        elif self.mode == "bi-linear":
            # score = self.bi_linear(input1=expand_query, input2=candidate_entity)
            score = torch.einsum("bh, hi, bci -> bc", query, self.bi_linear, candidate_entity)
        elif self.mode == "diag-bi-linear":
            score = torch.einsum("bh,h,bch->bc", query, self.diag_bi_linear, candidate_entity)
        else:
            raise ValueError(self.mode)

        # mask padding candidate score
        pad_mask = ~ candidate_attention_mask.bool()
        score: torch.Tensor = score.masked_fill(pad_mask, -10000.)

        return score  # without sigmoid


class ListwiseEmbeddings(nn.Module):
    def __init__(
            self,
            entity_emb_size: int,
            query_emb_size: int,
            use_query_hidden_size: bool,  # hidden_size同步为query_emb_size还是entity_emb_size
            max_position_emb: int,
            layer_norm_eps: float,
            hidden_dropout_prob: float,
            use_score_emb: bool,
            use_position_emb: bool,
    ):
        super().__init__()

        hidden_size = query_emb_size if use_query_hidden_size else entity_emb_size

        if entity_emb_size != hidden_size:
            self.entity_embedding_dense = nn.Linear(entity_emb_size, hidden_size, bias=False)
        else:
            self.entity_embedding_dense = nn.Identity()

        if query_emb_size != hidden_size:
            self.query_embedding_dense = nn.Linear(query_emb_size, hidden_size, bias=False)
        else:
            self.query_embedding_dense = nn.Identity()

        self.use_position_emb = use_position_emb
        self.use_score_emb = use_score_emb

        self.token_type_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)  # 2: query, doc
        self.entity_position_embeddings = nn.Embedding(num_embeddings=max_position_emb, embedding_dim=hidden_size) if self.use_position_emb else None
        self.score_embeddings = nn.Linear(1, hidden_size) if self.use_score_emb else None  # todo

        self.LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
            self,
            query: torch.Tensor,
            candidate_entity: torch.Tensor,
            candidate_entity_score: torch.Tensor,
    ):
        # query: [batch_size, query_emb_dim]
        # candidate_entity: [batch_size, candidate_num, entity_emb_dim]
        # candidate_entity_score: [batch_size, candidate_num]

        batch_size = query.shape[0]
        device = query.device
        candidate_num = candidate_entity.shape[1]

        # query
        query_embeddings = self.query_embedding_dense(query)  # [batch_size, hidden_size]
        query_token_type_ids = torch.zeros(size=(batch_size,), device=device, dtype=torch.long)  # [batch_size]
        query_token_type_embeddings = self.token_type_embeddings(query_token_type_ids)  # [batch_size, hidden_size]
        query_emb = query_embeddings + query_token_type_embeddings  # [batch_size, hidden_size]
        query_emb = query_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # candidate entity

        # entity emb
        candidate_entity_embeddings = self.entity_embedding_dense(candidate_entity)
        # [batch_size, candidate_num, hidden_size]

        # token_type emb (segment emb)
        candidate_token_type_ids = torch.ones(
            size=(candidate_entity.shape[0], candidate_entity.shape[1]), device=device, dtype=torch.long
        )  # [batch_size, candidate_num]
        candidate_token_type_embeddings = self.token_type_embeddings(candidate_token_type_ids)
        # [batch_size, candidate_num, hidden_size]

        # score emb
        candidate_score_embeddings = self.score_embeddings(candidate_entity_score.unsqueeze(-1)) \
            if self.use_score_emb else torch.zeros_like(candidate_entity_embeddings)
        # [batch_size, candidate_num, hidden_size]

        # position emb
        candidate_position_ids = torch.arange(0, candidate_num, device=device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, candidate_num]
        candidate_position_embeddings = self.entity_position_embeddings(candidate_position_ids) \
            if self.use_position_emb else torch.zeros_like(candidate_entity_embeddings)
        # [batch_size, candidate_num, hidden_size]

        # sum up
        candidate_emb = \
            candidate_entity_embeddings + \
            candidate_token_type_embeddings + \
            candidate_score_embeddings + \
            candidate_position_embeddings
        # [batch_size, candidate_num, hidden_size]

        # concat
        listwise_emb = torch.cat([
            query_emb,
            candidate_emb,
        ], dim=1)
        # [batch_size, 1+candidate_num, hidden_size]

        return listwise_emb


class IdentityPredictHead(nn.Module):
    """
    for debug
    """
    def __init__(
            self,
            config: BertConfig,
            entity_emb_size: int,
            max_position_emb: int,
            use_score_emb: bool,
            use_position_emb: bool,
            num_self_attention_layers: int,
            score_mode: str,
            score_fc_norm: bool,
            score_fc_xavier: bool,
    ):
        super().__init__()

    def forward(
            self,
            query: torch.Tensor,
            candidate_entity: torch.Tensor,
            candidate_entity_score: torch.Tensor,
            candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query: [batch_size, bert_hidden_size]
        # candidate_entity: [batch_size, candidate_num, entity_emb_dim]
        # candidate_entity_score: [batch_size, candidate_num]
        # candidate_attention_mask: [batch_size, candidate_num]

        return candidate_entity_score


class BatchNormListwisePredictHead(nn.Module):
    def __init__(
            self,
            config: BertConfig,
            entity_emb_size: int,
            max_position_emb: int,
            use_score_emb: bool,
            use_position_emb: bool,
            num_self_attention_layers: int,
            score_mode: str,
            score_fc_norm: bool,
            score_fc_xavier: bool,
            use_query_hidden_size: bool,
            norm_query_together: bool,
            affine: bool,  # True：有可学习的参数。（注：gamma、beta是可学习参数；均值、方差是统计出来的参数）
            track_running_stats: bool,  # True: 均值、方差是全局统计的；否则只是当前batch内统计的
    ):
        super().__init__()

        self.norm_query_together = norm_query_together

        # embedding
        query_emb_size = config.hidden_size
        self.listwise_embeddings = ListwiseEmbeddings(
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.hidden_dropout_prob,

            entity_emb_size=entity_emb_size,
            query_emb_size=query_emb_size,
            use_query_hidden_size=use_query_hidden_size,
            max_position_emb=max_position_emb,
            use_score_emb=use_score_emb,
            use_position_emb=use_position_emb,
        )

        listwise_config = BertConfig(**config.to_dict())
        listwise_config.hidden_size = query_emb_size if use_query_hidden_size else entity_emb_size
        self.listwise_config = listwise_config

        # batch norm
        self.batch_norm = nn.BatchNorm1d(
            num_features=listwise_config.hidden_size,
            eps=listwise_config.layer_norm_eps,
            affine=affine,
            track_running_stats=track_running_stats,
        )

        # score
        self.score_head = ListwiseScoreHead(
            mode=score_mode,
            fc_norm=score_fc_norm,
            fc_xavier=score_fc_xavier,
            feature_dim=listwise_config.hidden_size,
            dropout=listwise_config.hidden_dropout_prob,
        )

    def forward(
            self,
            query: torch.Tensor,
            candidate_entity: torch.Tensor,
            candidate_entity_score: torch.Tensor,
            candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query: [batch_size, bert_hidden_size]
        # candidate_entity: [batch_size, candidate_num, entity_emb_dim]
        # candidate_entity_score: [batch_size, candidate_num]
        # candidate_attention_mask: [batch_size, candidate_num]

        device = query.device

        # embedding
        batch_size = query.shape[0]
        listwise_emb = self.listwise_embeddings(
            query=query,
            candidate_entity=candidate_entity,
            candidate_entity_score=candidate_entity_score
        )  # [batch_size, 1+candidate_num, hidden_size]

        # batch norm
        # 注：虽然用的是batch norm，实际应该沿candidate方向norm，而不是真的沿"batch"方向
        if self.norm_query_together:
            # query和candidate一起norm
            listwise_emb_t = eop.rearrange(listwise_emb, 'b c h -> c h b')  # [1+candidate_num, hidden_size, batch_size]
            norm_listwise_emb_t = self.batch_norm(listwise_emb_t)  # [1+candidate_num, hidden_size, batch_size]
            listwise_seq_outputs = eop.rearrange(norm_listwise_emb_t, 'c h b -> b c h')  # [batch_size, 1+candidate_num, hidden_size]
            query_output = listwise_seq_outputs[:, 0, :]  # [batch_size, hidden_size]
            candidate_entity_output = listwise_seq_outputs[:, 1:, :]  # [batch_size, candidate_num, hidden_size]
        else:
            # 只norm candidate
            query_output = listwise_emb[:, 0, :]  # [batch_size, hidden_size]
            listwise_cand_emb_t = eop.rearrange(listwise_emb[:, 1:, :], 'b c h -> c h b')  # [candidate_num, hidden_size, batch_size]
            norm_listwise_cand_emb_t = self.batch_norm(listwise_cand_emb_t)  # [candidate_num, hidden_size, batch_size]
            candidate_entity_output = eop.rearrange(norm_listwise_cand_emb_t, 'c h b -> b c h')  # [batch_size, candidate_num, hidden_size]

        # score
        score = self.score_head(
            query=query_output,
            candidate_entity=candidate_entity_output,
            candidate_attention_mask=candidate_attention_mask,
        )

        # 注：只算打分，不把候选实体重新排序
        return score


class TransformerListwisePredictHead(nn.Module):
    def __init__(
            self,
            config: BertConfig,
            entity_emb_size: int,
            max_position_emb: int,
            use_score_emb: bool,
            use_position_emb: bool,
            num_self_attention_layers: int,
            score_mode: str,
            score_fc_norm: bool,
            score_fc_xavier: bool,
            use_query_hidden_size: bool,
    ):
        super().__init__()

        # embedding
        query_emb_size = config.hidden_size
        self.listwise_embeddings = ListwiseEmbeddings(
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.hidden_dropout_prob,

            entity_emb_size=entity_emb_size,
            query_emb_size=query_emb_size,
            use_query_hidden_size=use_query_hidden_size,
            max_position_emb=max_position_emb,
            use_score_emb=use_score_emb,
            use_position_emb=use_position_emb,
        )

        # encoder (self-attention)
        listwise_encoder_config = BertConfig(**config.to_dict())
        listwise_encoder_config.num_hidden_layers = num_self_attention_layers
        listwise_encoder_config.hidden_size = query_emb_size if use_query_hidden_size else entity_emb_size
        listwise_encoder_config.num_attention_heads = int(listwise_encoder_config.hidden_size / 64)  # 768 vs 12; 256 vs 4
        self.listwise_encoder_config = listwise_encoder_config
        self.encoder = BertEncoder(listwise_encoder_config)
        # self.pooler = BertPooler(config)

        # score
        self.score_head = ListwiseScoreHead(
            mode=score_mode,
            fc_norm=score_fc_norm,
            fc_xavier=score_fc_xavier,
            feature_dim=listwise_encoder_config.hidden_size,
            dropout=listwise_encoder_config.hidden_dropout_prob,
        )

    def forward(
            self,
            query: torch.Tensor,
            candidate_entity: torch.Tensor,
            candidate_entity_score: torch.Tensor,
            candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query: [batch_size, bert_hidden_size]
        # candidate_entity: [batch_size, candidate_num, entity_emb_dim]
        # candidate_entity_score: [batch_size, candidate_num]
        # candidate_attention_mask: [batch_size, candidate_num]

        device = query.device

        # embedding
        batch_size = query.shape[0]
        listwise_emb = self.listwise_embeddings(
            query=query,
            candidate_entity=candidate_entity,
            candidate_entity_score=candidate_entity_score
        )  # [batch_size, 1+candidate_num, hidden_size]

        # self attention
        query_attention_mask = torch.ones(size=(batch_size, 1), dtype=torch.long, device=device)  # [batch_size, 1]
        listwise_attention_mask = torch.cat([
            query_attention_mask,
            candidate_attention_mask,
        ], dim=1)  # [batch_size, 1+candidate_num]

        # 需要把1/0 mask转换为0./-10000.0 mask再过Encoder，luke、bert都是这么做的
        # 需要mask score时，采用的是score+mask的方式，HuggingFace说加法比mask_fill更高效
        listwise_attention_mask = self.get_extended_attention_mask(
            attention_mask=listwise_attention_mask,
            input_shape=listwise_attention_mask.shape,
            device=device,
        )
        listwise_encoder_outputs = self.encoder(
            listwise_emb,
            listwise_attention_mask,
            [None] * self.listwise_encoder_config.num_hidden_layers
        )  # 这是个len=1的tuple
        listwise_encoder_sequence_outputs = listwise_encoder_outputs[0]  # [batch_size, 1+candidate_num, hidden_size]

        query_output = listwise_encoder_sequence_outputs[:, 0, :]  # [batch_size, hidden_size]
        candidate_entity_output = listwise_encoder_sequence_outputs[:, 1:, :]  # [batch_size, candidate_num, hidden_size]

        # score
        score = self.score_head(
            query=query_output,
            candidate_entity=candidate_entity_output,
            candidate_attention_mask=candidate_attention_mask,
        )

        # 注：只算打分，不把候选实体重新排序
        return score

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device) -> torch.Tensor:
        """
        参考BertModel.get_extended_attention_mask

        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


