from typing import Union

from .blink_biencoder import BlinkBiEncoderRanker
from .toy_bert_huge_classifier import ToyBertHugeClassifier

UnionMyModel = Union[
    BlinkBiEncoderRanker,
    ToyBertHugeClassifier
]
