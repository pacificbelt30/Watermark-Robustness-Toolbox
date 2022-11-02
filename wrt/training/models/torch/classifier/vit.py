import mlconfig
import torchvision

import os

import numpy as np

from wrt.training.models.torch.classifier.tf_resnet import resnet101

# from models.modeling import CONFIGS, VisionTransformer
from ViT.models.modeling import CONFIGS, VisionTransformer

from torchinfo import summary


@mlconfig.register
def cifar_vit(dropout=0, **kwargs):
    model_type="ViT-B_16"
    pretrained_model = "./checkpoint/ViT-B_16.npz"
    vit = VisionTransformer(config=CONFIGS[model_type],img_size=kwargs['image_size'],num_classes=kwargs['num_classes'],zero_head=False,vis=False)
    summary(vit, (256,3,32,32),depth=4)
    vit.load_from(np.load(pretrained_model))
    return vit
