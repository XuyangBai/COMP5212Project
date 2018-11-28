#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:11:22 2018

@author: rongzhao
"""

import time
import torch
import torch.nn as nn
from torch import optim
from trainer import Trainer
from tensorboardX import SummaryWriter
import os
import os.path as P
import shutil
from model import ResNet18_Protein
from dataloader import get_data_loader, DataHub


device = torch.device('cuda:0')
timestr = time.strftime('%m%d%H%M')
this_fname = 'train.py'

verbose_output = False
imagenet = False
is_temp = False
is_small = False

if is_small:
    train_split, val_split, test_split = 'train-small', 'validation-small', 'test-small'
else:
    train_split, val_split, test_split = 'train', 'validation', 'test'
train_bs, test_bs = 256, 512
model_name = 'ResNet18_multitask_meta1'

data_root = '/home/rongzhao/projects/ml_kaggle_protein/data/npy'
data_kw = {
        'root': data_root,
        'train_bs': train_bs,
        'test_bs': test_bs,
        'train_sp': train_split,
        'val_sp': val_split,
        'test_sp': test_split,
        'mean': (13.528, 20.535, 14.249, 21.106),
        'std': (28.700, 38.161, 40.196, 38.172),
        'train_flip': (1, 1),
        'train_crop' : (384, 384), 
        'train_black': None, 
        'test_crop': None, 
        'num_workers': 4,
        }

data_cube = DataHub(**data_kw)

lr = 0.05
lr_scheme = {
        'base_lr': lr,
        'lr_policy': 'multistep',
#        'lr_policy': 'step', 
        'gamma': 0.3,
        'stepvalue': (100, 200, 300, ),
#        'stepsize': 1000,
        'max_epoch': 400,
        }

model = ResNet18_Protein(pretrain=imagenet)
if is_temp:
    experiment_id = '%s_temp' % model_name #_%s' % timestr
else:
    experiment_id = '%s_%s' % (model_name, timestr)
model_cube = {
        'model': model,
#        'init_func': misc.weights_init,
        'pretrain': '/home/rongzhao/projects/ml_kaggle_protein/snapshot/ResNet18_multitask_meta_11270113/state_500.pkl',
        'resume': None,
#        'optimizer': optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        'optimizer': optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
        }
criterion_cube = {
        'criterion': nn.BCEWithLogitsLoss(weight=None)
        }

snapshot_root = '../snapshot/%s' % (experiment_id)
os.makedirs(snapshot_root, exist_ok=True)
shutil.copy2(P.join('.', this_fname), P.join(snapshot_root, this_fname))

snapshot_scheme = {
        'root': snapshot_root,
        'display_interval': 1,
        'val_interval': 10,
        'snapshot_interval': 999999,
        }

writer = SummaryWriter(log_dir='../tboard/%s' % (experiment_id))
writer_cube = {
        'writer': writer,
        }

trainer = Trainer(model_cube, data_cube, criterion_cube, writer_cube, 
                  lr_scheme, snapshot_scheme, device)

train_m, val_m, test_m = trainer.train('f1_macro', verbose_output)
#train_m, val_m, test_m = trainer.test('/home/rongzhao/projects/ml_kaggle_protein/snapshot/ResNet18_multitask_11252204/state_600.pkl')


