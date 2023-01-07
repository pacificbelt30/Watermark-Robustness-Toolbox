from tqdm import tqdm
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
    parser.add_argument('-d', '--wm_dir', type=str, default='outputs/cifar10/wm/jia/00000',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--pretrained_dir", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    return parser.parse_args()

def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False

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

    def tsne(self,x:np.ndarray,y:np.ndarray,y_wm,layers:list):
        batch = 50
        # y = y[:len(x)//batch*batch]
        activation = np.array([])
        newy = []
        for i in tqdm(range(len(x)//batch)):
            tempx = x[i*batch:(i+1)*batch]
            all_activations = self.classifier.get_all_activations(tempx)
            if len(layers) == 0:
                layers = [i for i in range(len(all_activations))]
            activations = []
            for j,l in enumerate(layers):
                activations.append(all_activations[l])
            # activation = activations[-1].to('cpu').detach().numpy().copy()
            temp = []
            bias = 2
            for j in range(3):
                if j == 0:
                    temp = torch.flatten(activations[-j-bias],1).to('cpu').detach().numpy().copy()
                    continue
                temp = np.append(temp,torch.flatten(activations[-j-bias],1).to('cpu').detach().numpy().copy(),axis=1)
            if i == 0:
                activation = torch.flatten(activations[-bias],1).to('cpu').detach().numpy().copy()
                newy = torch.argmax(activations[-1],dim=1).to('cpu').detach().numpy().copy()
                # print(activation)
                activation = temp
                continue
            # print(torch.flatten(activations[-j-2],1).shape)
            # activation = np.append(activation,torch.flatten(activations[-bias],1).to('cpu').detach().numpy().copy(),axis=0)
            newy = np.append(newy,activations[-1].argmax(1).to('cpu').detach().numpy().copy(),axis=0)
            activation = np.append(activation,temp,axis=0)
        print(type(activation))
        print(newy[len(x)-len(y_wm):])
        print(y_wm==newy[len(x)-len(y_wm):])
        print('wm_acc:',sum(y_wm==newy[len(x)-len(y_wm):])/len(y_wm))
        
        tsne = TSNE(n_components=2, random_state=41)
        X_reduced = tsne.fit_transform(activation)
        plt.figure(figsize=(13, 7))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    c=y, cmap='Paired',
                    s=30, alpha=1.0)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.savefig('test.png')
        print('FINISH')


def main():
    args = parse_args()
    defense_config = mlconfig.load(args.wm_config)
    print(defense_config)
    pth_file = file_with_suffix_exists(dirname=args.wm_dir, suffix="best.pth")
    model_basedir, model_filename = os.path.split(pth_file)

    source_model: torch.nn.Sequential = defense_config.source_model()
    optimizer = defense_config.optimizer(source_model.parameters())

    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   filename=args.filename,
                                                   pretrained_dir=args.pretrained_dir)
    source_model.model.eval()
    source_model._model._model._return_hidden_activations = True
    # Load the training and testing data.
    # train_loader = defense_config.dataset(train=True)
    train_loader = defense_config.dataset(train=False)

    # Optionally load a dataset to load watermarking images from.
    wm_loader = None
    if "wm_dataset" in dict(defense_config).keys():
        wm_loader = defense_config.wm_dataset()
        print(f"Instantiated watermark loader (with {len(wm_loader)} batches): {wm_loader}")

    keylength = 100
    x,y = collect_n_samples(1000,train_loader)
    defense = defense_config.wm_scheme(classifier=source_model, optimizer=optimizer, config=defense_config)

    x_wm, y_wm = defense.load(filename=model_filename, path=model_basedir)
    # x_wm, y_wm = defense.load(filename='best.pth', path=model_basedir)
    # x_wm, y_wm = collect_n_samples(keylength, wm_loader, class_label=4, has_labels=True)
    x = np.append(x,x_wm,axis=0)
    y = np.append(y,np.array([10 for i in y_wm]))
    # y = np.append(y,y_wm)
    tsne = T_SNE(source_model)
    tsne.tsne(x,y,y_wm,[])

if __name__ == "__main__":
    main()
