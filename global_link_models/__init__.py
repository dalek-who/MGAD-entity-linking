from typing import Union

from .luke_link_local import LukeLinkLocal
from .luke_link_local_listwise import LukeLinkLocalListwise
from .entity_description_luke_link_local import EntityDescriptionLukeLinkLocal
from .entity_description_luke_link_local_with_teacher import EntityDescriptionLukeLinkLocalWithTeacher
from .luke_link_global_natural import LukeLinkGlobalNatural
from .luke_link_global_confidence import LukeLinkGlobalConfidence

UnionMyModel = Union[
    LukeLinkLocal,
    LukeLinkLocalListwise,
    EntityDescriptionLukeLinkLocal,
    EntityDescriptionLukeLinkLocalWithTeacher,
    LukeLinkGlobalNatural,
    LukeLinkGlobalConfidence,
]
