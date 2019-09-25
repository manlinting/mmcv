#!/usr/bin/env python
# encoding=utf8

from mmcv import Config

global cfg

def INIT_CONFIG(config_path):
    global cfg
    cfg = Config.fromfile(config_path)
    return cfg

def GET_CONFIG():
    return cfg
