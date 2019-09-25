#!/usr/bin/env python
#########################################################################
# Author: sofiawu
# Created Time: Wed 11 Oct 2017 02:40:40 PM CST
# File Name: imagelistfolder.py
# Description:
#########################################################################
import torch.utils.data as data
from PIL import Image
import os

def default_loader(path):
    return Image.open(path).convert("RGB")

class ImageListDataset(data.Dataset):
    def __init__(self, root_path, flist, transform=None, target_transform=None, loader=default_loader):
        self.imlist = []
        with open(flist, 'rb') as f:
            max_label = 0
            for line in f.readlines():
                filename, label = line.strip().split()
                if root_path != '':
                    filename = os.path.join(root_path, filename.decode())
                max_label = max(max_label, int(label))
                self.imlist.append((filename, int(label)))
            self.class_num = max_label+1
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = self.imlist

    def __getitem__(self, index):

        imline = self.imlist[index]
        img = self.loader(imline[0])
        if self.transform is not None:
            img = self.transform(img)

        return (img, imline[1])


    def __len__(self):
        return len(self.imlist)

    def get_class_num(self):
        return self.class_num
