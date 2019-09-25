#!/usr/bin/env python
# encoding=utf8
#########################################################################
# Author:
# Created Time: Thu 08 Nov 2018 08:48:39 PM CST
# File Name: convert.py
# Description:  tensor pil cv numpy
#########################################################################

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

# tensor [C, H, W] 取值范围是[0, 1.0] 一般经过normalization
# pil [H,W,C]  取值范围是[0,255]  RGB
# cv  [H,W,C] 取值范围是[0,255]  GBR

# pil to numpy
# np_obj = np.array( pil_obj )

# numpy to pil i
# pil_obj = PIL.Image.fromarray( np_obj ).convert('RGB')

# tensor => numpy
# np_obj = tensor.numpy()

# numpy => tensor
# tensor = torch.Tensor(np_obj)

# pil to cv
# cv_obj = np.array(pil_img)[:, :, ::-1].copy()

# cv to pil
# pil_obj = PIL.Image.fromarray(cv_obj.astype('uint8')[:, :, ::-1], mode='RGB')

# tensor to pil
# pil_img = transforms.ToPILImage()(tensor_obj).convert("RGB")
# = transpose + *255

def tensor_to_pil(tensor_img, MEAN=[], STD=[]):
    if MEAN and STD:
        np_img = tensor_img.numpy()
        for i in range(0, 3):
            np_img[i] = np_img[i] * STD[i] + MEAN[i]  # unnormalize
        pil_img = transforms.ToPILImage()(torch.from_numpy(np_img)).convert("RGB")
    else:
        pil_img = transforms.ToPILImage()(tensor_img).convert("RGB")
    return pil_img

def tensor_to_cv(tensor_img, MEAN=[], STD=[]):
    pil_img = tensor_to_pil(tensor_img, MEAN, STD)
    cv_img = np.array(pil_img)[:, :, ::-1].copy()
    return cv_img


if __name__ == '__main__':

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    pil_img = PIL.Image.open("color.jpg").convert("RGB")

    # pil to tensor
    tensor_img = img_transform(pil_img)

    pil_img = tensor_to_pil(tensor_img, MEAN, STD)
    pil_img.save("pil.jpg")
    cv_img = np.array(pil_img)[:, :, ::-1].copy()
    cv2.imwrite("cv.jpg", cv_img)
