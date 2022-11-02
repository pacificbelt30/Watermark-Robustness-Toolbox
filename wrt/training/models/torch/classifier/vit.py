import mlconfig
import torchvision

import os

from wrt.training.models.torch.classifier.tf_resnet import resnet101

# from models.modeling import CONFIGS, VisionTransformer
from ViT-pytorch.models.modeling import CONFIGS, VisionTransformer


@mlconfig.register
def cifar_vit(dropout=0, **kwargs):
    model_type="ViT-B_16"
    return VisionTransformer(config=CONFIGS[model_type],img_size=kwargs['image_size'],num_classes=kwargs['num_classes'],zero_head=False,vis=False)
