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
    vit = VisionTransformer(config=CONFIGS[model_type],img_size=kwargs['image_size'],num_classes=kwargs['num_classes'],zero_head=True,vis=False,only_logits=True)
    summary(vit, (256,3,kwargs['image_size'],kwargs['image_size']),depth=4)
    if kwargs['pretrained_model']:
        vit.load_from(np.load(pretrained_model))
    return vit
