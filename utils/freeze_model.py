from typing import List, Dict, Tuple, Union, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


def freeze(model: Union[nn.Module, pl.LightningModule, nn.Parameter]):
    if isinstance(model, nn.Parameter):
        model.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


def un_freeze(model: Union[nn.Module, pl.LightningModule, nn.Parameter]):
    if isinstance(model, nn.Parameter):
        model.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
