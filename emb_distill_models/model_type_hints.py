# helps to type hints
from typing import Union, List, Tuple, Dict, NamedTuple, Optional
from typing_extensions import Protocol
from dataclasses import dataclass, field
import torch


@dataclass()
class BaseModelReturn:
    # 注：这里不要添加默认值，否则如果子类继承BaseModelReturn，额外添加的属性也必须要添加默认值
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    wikipedia_pageid: torch.Tensor
    student_entity_emb: torch.Tensor
    student_entity_bias: torch.Tensor
    batch_size: int

    loss_student_link: torch.Tensor  # student在link任务上的loss
    loss_score_distill: torch.Tensor  # teacher score指导student score的loss
    loss_emb_ce: torch.Tensor  # 拟合实体embedding的loss
    loss_emb_mse: torch.Tensor  # 拟合实体embedding的loss
    loss_emb_invert_ce: torch.Tensor  # 拟合实体embedding的loss
    loss_bias: torch.Tensor  # 拟合实体bias的loss

    loss_total: torch.Tensor  # 总loss，用来求导

    # 从global link借鉴的
    student_label: Optional[torch.Tensor] = None
    student_predict_score: Optional[torch.Tensor] = None
    student_out_topk_predict_score: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    student_out_topk_indices: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    student_mention_emb: Optional[torch.Tensor] = None

