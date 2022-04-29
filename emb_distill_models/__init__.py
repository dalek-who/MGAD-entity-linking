from typing import Union

from .feature_2_emb import Feature2Emb
from .luke_teacher_student import LukeTeacherStudent
from .entity_discription_luke import \
    EntityDescriptionLukeForEntityEncode, \
    EntityDescriptionLukeForMentionEncode
from .emb_distill_teacher_student import EmbDistillTeacherStudent

UnionMyModel = Union[
    Feature2Emb,
    LukeTeacherStudent,
    EmbDistillTeacherStudent,
    EntityDescriptionLukeForMentionEncode,
    EntityDescriptionLukeForEntityEncode,
]
