from bcos.models.resnet import resnet18, resnet34, resnet50, BcosResNet
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

def make_resnet34(
    logger: logging.Logger, 
    pretrained: bool = False, 
    **kwargs
) -> EnhancedResNet:
    model = resnet34(pretrained=pretrained, **kwargs)
    model.__class__ = EnhancedResNet 
    model.logger = logger
    return model


def make_resnet50(
    logger: logging.Logger, 
    pretrained: bool = False, 
    **kwargs
) -> EnhancedResNet:
    model = resnet50(pretrained=pretrained, **kwargs)
    model.__class__ = EnhancedResNet 
    model.logger = logger
    return model
