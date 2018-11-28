#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:44:04 2018

@author: rongzhao
"""

import torch
import torch.nn as nn
from torchvision import models
import time

def weights_init(m):
    '''
    ConvXd: kaiming_normal (weight), zeros (bias)
    BatchNormXd: ones (weight), zeros (bias)
    '''
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def resnet18_protein(in_stride=4, pretrain=False, in_features=4, out_features=28):
    '''Construct a network based on ResNet18'''
    model = models.resnet18(pretrain)
    model.conv1 = nn.Conv2d(in_features, 64, (9,9), (in_stride,in_stride), (4,4), bias=False)
    model.fc = nn.Linear(512, out_features, bias=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    if not pretrain:
        model.apply(weights_init)
        
    return model

def f1_loss(y_true, y_pred):
    
    tp = torch.sum( (y_true*y_pred).float(), dim=0)
    tn = torch.sum( ((1-y_true)*(1-y_pred)).float(), dim=0)
    fp = torch.sum( ((1-y_true)*y_pred).float(), dim=0)
    fn = torch.sum( (y_true*(1-y_pred)).float(), dim=0)

    epsilon = 1e-07
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2*p*r / (p+r+epsilon)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)
    
#t = time.time()
#v = torch.ones(128, 4, 512, 512)
#v = v.cuda()
#v.normal_(0, 0.5)
#print('Finish loading data to GPU at %.2fs' % (time.time()-t))
#m = resnet18_protein(pretrain=True)
#m.cuda()
#print('Start computing at %.2fs' % (time.time()-t))
#y = m(v)
#print('Finish computing at %.2fs' % (time.time()-t))
