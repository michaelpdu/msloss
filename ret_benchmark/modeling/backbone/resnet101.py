from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.models as models
from ret_benchmark.modeling import registry
from ret_benchmark.modeling.backbone.resnet import ResNet


@registry.BACKBONES.register('resnet101')
class ResNet101(ResNet):

    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=False)
        self.freeze_layers()
