
import argparse
import json
import os
import time
from datetime import datetime
from shutil import copyfile

import mlconfig
import torch

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from wrt.classifiers.pytorch import PyTorchClassifier
from wrt.training.datasets.utils import collect_n_samples

from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_config', type=str, default='configs/cifar10/wm_configs/dawn1.yaml',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--pretrained_dir", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    return parser.parse_args()

def __load_model(model, optimizer, image_size, num_classes, pretrained_dir: str = None,
                 filename: str = 'best.pth'):
    """ Loads a (pretrained) source model from a directory and wraps it into a PyTorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    if pretrained_dir:
        assert filename.endswith(".pth"), "Only '*.pth' are allowed for pretrained models"
        print(f"Loading a pretrained source model from '{pretrained_dir}'.")
        state_dict = torch.load(os.path.join(pretrained_dir, filename))
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model

class T_SNE():
    def __init__(self,model:PyTorchClassifier):
        self.classifier = model

    def tsne(self,x:np.ndarray,y:np.ndarray,layers:list):
        all_activations = self.classifier.get_all_activations(x)
        if len(layers) == 0:
            layers = [i for i in range(len(all_activations))]
        activations = []
        for i,l in enumerate(layers):
            activations.append(all_activations[l])
        activation = activations[-1]
        
        tsne = TSNE(n_components=2, random_state=41)
        X_reduced = tsne.fit_transform(activation)
        plt.figure(figsize=(13, 7))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    c=y, cmap='jet',
                    s=15, alpha=0.5)
        plt.axis('off')
        plt.colorbar()
        plt.show()


def main():
    args = parse_args()
    wm_config = 'configs/cifar10/wm_configs_experiments/jia.yaml'
    defense_config = mlconfig.load(wm_config)
    print(defense_config)

    source_model: torch.nn.Sequential = defense_config.source_model()
    optimizer = defense_config.optimizer(source_model.parameters())

    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   filename=args.filename,
                                                   pretrained_dir=args.pretrained_dir)
    # Load the training and testing data.
    train_loader = defense_config.dataset(train=True)
    valid_loader = defense_config.dataset(train=False)

    # Optionally load a dataset to load watermarking images from.
    wm_loader = None
    if "wm_dataset" in dict(defense_config).keys():
        wm_loader = defense_config.wm_dataset()
        print(f"Instantiated watermark loader (with {len(wm_loader)} batches): {wm_loader}")

    x,y = collect_n_samples(100,train_loader)
    tsne = T_SNE(source_model)
    tsne.tsne(x,y,[])

if __name__ == "__main__":
    main()
