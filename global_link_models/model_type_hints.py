# helps to type hints
from typing import Union, List, Tuple, Dict, NamedTuple, Optional
from typing_extensions import Protocol
from dataclasses import dataclass, field
import torch


@dataclass()
class BaseModelReturn:
    # 注：这里不要添加默认值，否则如果子类继承BaseModelReturn，额外添加的属性也必须要添加默认值
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    query_id: List[str]
    has_gold: List[bool]
    freq_bin: List[int]
    gold_wikipedia_pageid: List[int]
    gold_wikidata_id: List[str]
    label: torch.Tensor
    predict_score: torch.Tensor
    out_topk_predict_score: Union[torch.Tensor, List[torch.Tensor]]
    out_topk_indices: Union[torch.Tensor, List[torch.Tensor]]
    loss: Optional[torch.Tensor]

    mention_emb: Optional[torch.Tensor] = None
    entity_emb: Optional[torch.Tensor] = None

    loss_score_distill: Optional[torch.Tensor] = None
    loss_link: Optional[torch.Tensor] = None

