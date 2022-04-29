import logging
from typing import List, Dict, Tuple, NamedTuple, Union, Optional
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities import move_data_to_device
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


from link_dataset import ModelExample, BertInput
from link_models.model_type_hints import BaseModelReturn


class ToyBertHugeClassifier(pl.LightningModule):
    def __init__(
            self,
            pretrained_dir: Union[Path, str],
            tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast,
            in_topk_candidates: Union[int, str],
            out_topk_candidates: int,
            max_context_len: int,
            max_entity_len: int,
            max_seq_len: int,
            mention_add_start_end: bool,
            model_data_format: str,
            out_dim: int,
            pull_from_layer,
            add_linear: bool,
            entity_vocab_table_name: str,
            seed: int,
            shared=None,
            py_logger: logging.Logger=None,
    ):
        super().__init__()
        self.out_topk_candidates = out_topk_candidates

        self.bert = AutoModel.from_pretrained(pretrained_dir)
        self.test_classifier = nn.Linear(self.bert.config.hidden_size, 1_000_000)

        self.main_device: Optional[torch.device] = None
        self.device_backbone = torch.device("cuda:0")  # cuda:0 is main device
        self.device_classifier = torch.device("cuda:1")

    def split_model_to_devices(self):  # todo, maybe call inside a callback, like on_train_start, on_test_start?
        # 如果在trainer指定gpus=1，且在on_xxx_epoch_start时调用该函数，那么在调用此函数时，model已经被分配到同一个cuda上了。
        # 但是因为还没forward，仅仅一个模型的尺寸，单卡还是能放下的，在forward之前分配到各张卡上即可

        if next(self.bert.parameters()).device != self.device_backbone:
            self.bert.to(self.device_backbone)
        if next(self.test_classifier.parameters()).device != self.device_classifier:
            self.test_classifier.to(self.device_classifier)
        self.main_device = self.device_backbone
        pass

    def forward(self, batch: ModelExample):
        if self.main_device is not None:  # None means split_model_to_devices is not called, and batch has been auto moved
            batch = move_data_to_device(batch=batch, device=self.main_device)

        # 这里一边传输，下面bert一边计算，流水线加速
        label = batch.label.to(self.device_classifier) \
            if batch.label.max().item() < self.test_classifier.weight.shape[0] \
            else torch.randint(low=0, high=self.test_classifier.weight.shape[0], size=batch.label.shape,
                               device=self.device_classifier)  # fake label, just for test

        bert_result: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            batch.q_text.input_ids,
            batch.q_text.token_type_ids,
            batch.q_text.attention_mask,
        )
        scores = self.test_classifier(bert_result.pooler_output.to(self.device_classifier))
        topk_values, topk_indices = scores.topk(k=self.out_topk_candidates, dim=1)
        loss = F.cross_entropy(input=scores, target=label)
        result = BaseModelReturn(
            predict_score=scores,
            out_topk_predict_score=topk_values,
            out_topk_indices=topk_indices,
            loss=loss,
        )
        return result

