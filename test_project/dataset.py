#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

import torch
import torch.backends.cudnn
import torchvision
import PIL
import numpy as np

from mmcv import config
from utils.dataset.dataset_tar import DatasetTar
from utils.data import collate
import pickle

def random_lighting(img: PIL.Image.Image, std: float) -> PIL.Image.Image:
    eigval = torch.tensor([0.2175, 0.0188, 0.0045]) * 255.0
    eigvec = torch.tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
        ])
    rgb = eigvec.matmul(std * torch.randn(3) * eigval)
    img = torch.tensor(np.array(img)).float() + rgb
    img = img.round().clamp(0, 255).numpy().astype(np.uint8)
    return PIL.Image.fromarray(img)

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def init_loader_train():
        def process_fn(name, data, info):
            image = PIL.Image.open(io.BytesIO(data)).convert('RGB')
            image = transform_fn(image)
            label = info
            return image, label

        transform_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(cfg.data.train.image_size, (0.64, 1.0)),
            torchvision.transforms.RandomOrder([
                torchvision.transforms.ColorJitter(0.2, 0, 0),
                torchvision.transforms.ColorJitter(0, 0.2, 0),
                torchvision.transforms.ColorJitter(0, 0, 0.2),
                lambda img: random_lighting(img, 0.1),
            ]),

            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=cfg.data.train.mean,
                std=cfg.data.train.std,
            ),
        ])
    dataset = DatasetTar(
            cfg.data.train.path, process_fn)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.train.batch_size,
            shuffle=cfg.data.train.shuffle,
            num_workers=cfg.data.train.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate(x, 'tl'),
            )
    return loader


def init_loader_val():
    def process_fn( name, data, info ):
        image = PIL.Image.open(io.BytesIO(data)).convert('RGB')
        image = transform_fn(image)
        label = info
        return image, label

    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(cfg.data.val.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=cfg.data.val.mean,
            std=cfg.data.val.std,
            ),
        ])
    dataset = DatasetTar(
            cfg.data.val.path, process_fn)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.val.batch_size,
            shuffle=cfg.data.val.shuffle,
            num_workers=cfg.data.val.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate(x, 'tl'),
            )
    return loader
