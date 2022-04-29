# helps to type hints
from typing import Union, List, Tuple, Dict, NamedTuple, Optional
from typing_extensions import Protocol
from dataclasses import dataclass, field
import torch


@dataclass()
class BaseModelReturn:
    # 注：这里不要添加默认值，否则如果子类继承BaseModelReturn，额外添加的属性也必须要添加默认值
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    predict_score: torch.Tensor
    out_topk_predict_score: torch.Tensor
    out_topk_indices: torch.Tensor
    loss: Optional[torch.Tensor]
