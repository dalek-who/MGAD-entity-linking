from typing import Tuple, NamedTuple, Type, Union, List, Dict
import torch
from typing_inspect import is_union_type


def issubclass_namedtuple(class_type):
    return issubclass(class_type, Tuple) and hasattr(class_type, "_fields")

def is_optional_type(class_type):
    return \
        hasattr(class_type, "__args__") \
        and len(class_type.__args__) == 2 \
        and class_type.__args__[-1] is type(None)

def extract_optional_type(class_type):
    assert is_optional_type(class_type)
    return class_type.__args__[0]

def isinstance_namedtuple(obj):
    return isinstance(obj, Tuple) and hasattr(obj, "_fields")

def fill_into_namedtuple(data: Tuple, tuple_class: Type[Union[Tuple, NamedTuple]]):
    assert not is_union_type(tuple_class) and issubclass_namedtuple(tuple_class)  # NamedTuple只是个语法糖，不是类
    new_data_dict = dict()
    for field_data, (field_name, field_type) in zip(data, tuple_class._field_types.items()):
        if is_optional_type(field_type):
            field_type = extract_optional_type(field_type)
        if not is_union_type(field_type) and issubclass_namedtuple(field_type) and field_data is not None:
            new_data_dict[field_name] = fill_into_namedtuple(data=field_data, tuple_class=field_type)
        else:
            new_data_dict[field_name] = field_data
    new_data = tuple_class(**new_data_dict)
    return new_data


if __name__ == '__main__':
    class BertInput(NamedTuple):
        # type hint: first: single_format; second: batch_format
        input_ids: Union[List[int], torch.Tensor]
        token_type_ids: Union[List[int], torch.Tensor] = None
        attention_mask: Union[List[int], torch.Tensor] = None


    class ModelExample(NamedTuple):
        """
        candidate是纯文本形式
        """
        # type hint: first: single_format; second: batch_format
        wikipedia_pageid: Union[int, torch.Tensor]
        entity_text: BertInput
        teacher_emb: torch.Tensor
        teacher_bias: torch.Tensor


    seq_len = 50
    batch_size = 7
    hidden_size = 256
    wikipedia_pageid = list(range(2, 2+batch_size))
    entity_text = (
        torch.ones(size=(batch_size, seq_len)) * 3,
        torch.zeros(size=(batch_size, seq_len)),
        torch.ones(size=(batch_size, seq_len)),
    )
    teacher_emb = torch.rand(size=(batch_size, hidden_size))
    teacher_bias = torch.rand(size=(batch_size, ))
    model_example = (
        wikipedia_pageid,
        entity_text,
        teacher_emb,
        teacher_bias,
    )
    t = fill_into_namedtuple(data=model_example, tuple_class=ModelExample)
    pass

