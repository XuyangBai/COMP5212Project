#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:54:24 2018

@author: rongzhao
"""

import time
import torch
import torch.nn as nn
from torch import optim
from trainer_meta import Trainer_Meta
from tensorboardX import SummaryWriter
import os
import os.path as P
import shutil
from model import ResNet18_Protein
from dataloader import get_data_loader, DataHub


device = torch.device('cuda:0')
timestr = time.strftime('%m%d%H%M')
this_fname = 'train_meta.py'

verbose_output = False
imagenet = False
is_temp = False
is_small = False

if is_small:
    train_split, val_split, test_split = 'train-small', 'validation-small', 'test-small'
else:
    train_split, val_split, test_split = 'train', 'validation', 'test'
train_bs, test_bs = 256, 512
model_name = 'ResNet18_multitask_meta'

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

lr = 1
lr_scheme = {
        'base_lr': lr,
        'lr_policy': 'multistep',
        'gamma': 0.3,
        'stepvalue': (250, 400, 500, ),
        'max_epoch': 600,
        }
lr_inner = 0.1
lr_scheme_inner = {
        'base_lr': lr_inner,
        'lr_policy': 'multistep',
        'gamma': 0.3,
        'stepvalue': (250, 400, 500, ),
        'max_epoch': 600,
        }
lr_cube = {
        'lr_scheme': lr_scheme,
        'lr_scheme_inner': lr_scheme_inner,
        'k': 5,
        }

model = ResNet18_Protein(pretrain=imagenet)
model_inner = ResNet18_Protein(pretrain=imagenet)
if is_temp:
    experiment_id = '%s_temp' % model_name #_%s' % timestr
else:
    experiment_id = '%s_%s' % (model_name, timestr)
model_cube = {
        'model': model,
        'model_inner': model_inner,
#        'init_func': misc.weights_init,
        'pretrain': None,
        'resume': None,
#        'optimizer': optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        'optimizer': optim.SGD(model.parameters(), lr=lr),
        'optimizer_inner': optim.SGD(model_inner.parameters(), lr=lr_inner, weight_decay=5e-4),
        }
criterion_cube = {
        'criterion': nn.BCEWithLogitsLoss(weight=None)
        }

snapshot_root = '../snapshot/%s' % (experiment_id)
os.makedirs(snapshot_root, exist_ok=True)
shutil.copy2(P.join('.', this_fname), P.join(snapshot_root, this_fname))

snapshot_scheme = {
        'root': snapshot_root,
        'display_interval': 10,
        'val_interval': 10,
        'snapshot_interval': 999999,
        }

writer = SummaryWriter(log_dir='../tboard/%s' % (experiment_id))
writer_cube = {
        'writer': writer,
        }

trainer = Trainer_Meta(model_cube, data_cube, criterion_cube, writer_cube, 
                  lr_cube, snapshot_scheme, device)

train_m, val_m, test_m = trainer.train('f1_macro', verbose_output)
#train_m, val_m, test_m = trainer.test('/home/rongzhao/projects/ml_kaggle_protein/snapshot/ResNet18_multitask_11252204/state_600.pkl')


