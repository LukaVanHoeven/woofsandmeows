from bcos.models.resnet import resnet18, BcosResNet
from base_model import BaseModel

import logging


class EnhancedResNet(BaseModel, BcosResNet):
    pass

def make_resnet18(
    logger: logging.Logger, 
    pretrained: bool = False, 
    **kwargs
) -> EnhancedResNet:
    model = resnet18(pretrained=pretrained, **kwargs)
    model.__class__ = EnhancedResNet 
    model.logger = logger
    return model
