#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:44:04 2018

@author: rongzhao
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable as V
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

class ResNet18_Protein(nn.Module):
    '''This is a wrap of resnet18 for Meta-Learning'''
    def __init__(self, in_stride=4, pretrain=False, in_features=4, out_features=28):
        super(ResNet18_Protein, self).__init__()
        self.init_kw = {'in_stride': in_stride, 'pretrain': pretrain, 
                        'in_features': in_features, 'out_features': out_features}
        self.net = resnet18_protein(in_stride, pretrain, in_features, out_features)
        
    def forward(self, x):
        return self.net(x)
    
    def copy(self, device, mode='train'):
        model = resnet18_protein(**self.init_kw)
        model.to(device)
        if mode == 'train':
            model.train()
        elif mode in ('eval', 'test'):
            model.eval()
        model.load_state(self.net.state_dict())
        return model
    
    def copy_state(self, src_model):
        self.load_state_dict(src_model.state_dict())
        
    def accum_grad(self, src_model, k, lr_inner):
        name_to_param = dict(self.named_parameters())
        for name, param in src_model.named_parameters():
            cur_grad = (param.grad.detach()) / k / lr_inner
            if name_to_param[name].grad is None:
                name_to_param[name].grad = torch.zeros(cur_grad.size())
            name_to_param[name].grad.detach().add_(cur_grad)
    
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
