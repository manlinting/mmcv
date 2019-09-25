#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import functools

import torch
import torch.nn as nn

def load_parameters(model, src_state_dict):
    logging.info('Loading Parameters...')
    if isinstance(src_state_dict, str):
        src_state_dict = torch.load(
            src_state_dict, map_location=lambda storage, loc: storage)
    dst_state_dict = model.state_dict()
    for k in dst_state_dict:
        if k in src_state_dict:
            if src_state_dict[k].size() == dst_state_dict[k].size():
                # logging.info('{}: Loaded.'.format(k))
                dst_state_dict[k] = src_state_dict[k]
            else:
                logging.warning('{}: Ignored due to shapes.'.format(k))
        else:
            logging.warning('{}: Ignored due to missing.'.format(k))
    model.load_state_dict(dst_state_dict)


def get_num_parameters(net):
    parameters = net.state_dict()
    return functools.reduce(lambda x, y: x + y,
                            [parameters[x].numel() for x in parameters])


def get_num_flops(net, x):
    def forward_hook(m, input, output):
        if type(m) == nn.Linear:
            output_size = torch.tensor(output[0].shape)
            flops = m.weight.numel()
            if m.bias is not None:
                flops += m.bias.numel()
            m.flops = flops
            return
        if type(m) in [
                nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                nn.ConvTranspose2d, nn.ConvTranspose3d
        ]:
            output_size = torch.tensor(output[0].shape)
            flops = m.weight.numel() * output_size[1:].prod() / m.groups
            if m.bias is not None:
                flops += m.bias.numel() * output_size[1:].prod()
            m.flops = flops
            return
        if type(m) in [
                nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d,
                nn.AvgPool2d, nn.AvgPool3d
        ]:
            output_size = torch.tensor(output[0].shape)
            kernel_size = m.kernel_size
            if type(kernel_size) not in (tuple, list):
                kernel_size = [kernel_size] * (len(output_size) - 1)
            m.flops = torch.tensor(kernel_size).prod() * output_size[1:].prod()
            return
        if type(m) in [nn.ReLU, nn.ReLU6, nn.PReLU, nn.Sigmoid]:
            output_size = torch.tensor(output[0].shape)
            m.flops = output_size.prod()
            return
        if type(m) in [
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
        ]:
            output_size = torch.tensor(output[0].shape)
            m.flops = output_size.prod() * 4 if m.affine else 2
            return

    assert x.shape[0] == 1
    handles = list(
        map(lambda x: x.register_forward_hook(forward_hook), net.modules()))
    with torch.no_grad():
        net(x)
    list(map(lambda x: x.remove(), handles))
    return sum([x.flops for x in net.modules() if hasattr(x, 'flops')])
