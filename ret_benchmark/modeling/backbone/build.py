from ret_benchmark.modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet50 import ResNet50
from .resnet101 import ResNet101
from .resnet152 import ResNet152


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME in BACKBONES, \
        f"backbone {cfg.MODEL.BACKBONE} is not registered in registry : {BACKBONES.keys()}"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME]()
