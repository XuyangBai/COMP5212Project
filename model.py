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
import math

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

class DenseNet_Protein(nn.Module):
    '''This is a wrap of resnet18 for Meta-Learning'''
    def __init__(self, in_stride=4, depths=8, in_features=4, out_features=28, growth_rate=16, 
                 fc_drop=None, bottle_neck=None):
        super(DenseNet_Protein, self).__init__()
        self.init_kw = {'in_stride': in_stride, 'fc_drop': fc_drop, 'growth_rate':growth_rate,
                        'in_features': in_features, 'out_features': out_features,
                        'bottle_neck': bottle_neck, 'depths': depths}
        self.net = densenet(**self.init_kw)
        
    def forward(self, x):
        return self.net(x)
    
    def copy(self, device, mode='train'):
        model = densenet(**self.init_kw)
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

def passthrough(x):
    return x

class PassModule(nn.Module):
    def __init__(self):
        super(PassModule, self).__init__()
    def forward(self, x):
        return x

class DenseBlock(nn.Module):
    def __init__(self, nIn, growth_rate, depth, drop_rate=0, only_new=False,
                 bottle_neck=None, is_bn=True):
        super(DenseBlock, self).__init__()
        self.only_new = only_new
        self.depth = depth
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList([self.get_transform(
            nIn + i * growth_rate, growth_rate, bottle_neck,
            drop_rate, is_bn) for i in range(depth)])

    def forward(self, x):
        if self.only_new:
            outputs = []
            for i in range(self.depth):
                tx = self.layers[i](x)
                x = torch.cat((x, tx), 1)
                outputs.append(tx)
            return torch.cat(outputs, 1)
        else:
            for i in range(self.depth):
                x = torch.cat((x, self.layers[i](x)), 1)
            return x

    def get_transform(self, nIn, nOut, bottle_neck=None, drop_rate=0, is_bn=True):
        if not bottle_neck:
            if is_bn:
                return nn.Sequential(
                    nn.BatchNorm2d(nIn),
                    nn.ReLU(True),
                    nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=False),
                    nn.Dropout2d(drop_rate),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=False),
                    nn.Dropout2d(drop_rate),
                )
        else:
            nBottle = bottle_neck
            if is_bn:
                return nn.Sequential(
                    nn.BatchNorm2d(nIn),
                    nn.ReLU(True),
                    nn.Conv2d(nIn, nBottle, 1, stride=1, padding=0, bias=False),
                    nn.Dropout2d(drop_rate),
                    nn.BatchNorm2d(nBottle),
                    nn.ReLU(True),
                    nn.Conv2d(nBottle, nOut, 3, stride=1, padding=1, bias=False),
                    nn.Dropout2d(drop_rate),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(nIn, nBottle, 1, stride=1, padding=0, bias=False),
                    nn.Dropout2d(drop_rate),
                    nn.BatchNorm2d(nBottle),
                    nn.ReLU(True),
                    nn.Conv2d(nBottle, nOut, 3, stride=1, padding=1, bias=False),
                    nn.Dropout2d(drop_rate),
                )


class Dense2D(nn.Module):
    def __init__(self, nMods, growth_rates, depths, n_scales=3, compress=1, 
                 fc_drop=None, n_channel_start=32, nClass=28, drop_rate=0.0, 
                 init_stride=1, bottle_neck=False):
        super(Dense2D, self).__init__()
        self.depths = [depths] * n_scales if type(depths) == int else depths
        self.growth_rates = [growth_rates] * n_scales \
                            if type(growth_rates) == int else growth_rates
        assert len(self.depths) == n_scales, 'Inconsistent depths'
        assert len(self.growth_rates) == n_scales, 'Inconsistent growth_rates'

        self.n_scales = n_scales
        self.init_stride = init_stride
        inconv_t = [nn.Conv2d(nMods, n_channel_start, 9, init_stride, 4, bias=False),
                    nn.BatchNorm2d(n_channel_start),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(n_channel_start, n_channel_start, 1, 1, 0, bias=False),]
        self.inconv = nn.Sequential(*inconv_t)
        self.dense_blocks = nn.ModuleList([])
        self.transition_downs = nn.ModuleList([])
        self.fc_deconv = nn.ModuleList([])
        if self.init_stride != 1:
            self.last_deconv = nn.ModuleList([])

        stride = []
        for _ in range(n_scales - 1):
            stride.append(2)

        # Dense Blocks
        nIn = n_channel_start
        nskip = []
        for i in range(n_scales):
            self.dense_blocks.append(
                    DenseBlock(nIn, self.growth_rates[i], self.depths[i],
                               drop_rate, bottle_neck=bottle_neck))
            nIn += self.growth_rates[i] * self.depths[i]
            nskip.append(nIn)
            nIn = math.floor(compress * nIn)

        # Transition Down
        for i in range(n_scales-1):
            nIn = nskip[i]
            nInC = math.floor(nIn * compress)
            t_td = [nn.BatchNorm2d(nIn), nn.ReLU(True),
                    nn.Conv2d(nIn, nInC, 1,1,0, bias=False),
                    nn.AvgPool2d(stride[i], stride[i], padding=0)]
            self.transition_downs.append(nn.Sequential(*t_td))
        # After the last dense block
        t_td = [nn.BatchNorm2d(nskip[-1]),
                nn.AdaptiveAvgPool2d((1,1))]
        self.transition_downs.append(nn.Sequential(*t_td))
        self.fc = nn.Linear(nskip[-1], nClass, True)
        
    def forward(self, x):
        out = self.inconv(x)
        for i in range(self.n_scales):
            out = self.dense_blocks[i](out)
            out = self.transition_downs[i](out)
        out.squeeze_(dim=2).squeeze_(dim=2)
        out = self.fc(out)
        return out
################################ n_scale = 4 ################################
def densenet(in_stride=4, depths=8, in_features=4, out_features=28, growth_rate=16, 
             fc_drop=None, bottle_neck=None):
    model = Dense2D(nMods=in_features, growth_rates=growth_rate, depths=depths, 
                    n_scales=4, compress=0.5, n_channel_start=64, 
                    init_stride=in_stride, fc_drop=fc_drop, 
                    nClass=out_features, bottle_neck=bottle_neck)
    return model











