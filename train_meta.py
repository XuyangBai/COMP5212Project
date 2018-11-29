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
from trainer import Trainer
from tensorboardX import SummaryWriter
import os
import os.path as P
import numpy as np
import shutil
from utils import get_primitive_prob, normalize
from model import ResNet18_Protein
from dataloader import get_data_loader, DataHub


device = torch.device('cuda:0')
timestr = time.strftime('%m%d%H%M')
this_fname = 'train_meta.py'

verbose_output = False
imagenet = False
first14 = False
binary_meta = False
is_temp = False
is_small = False

if is_small:
    train_split, val_split, test_split = 'train-small', 'validation-small', 'test-small'
    train_bs, test_bs = 16, 16
else:
    train_split, val_split, test_split = 'train', 'validation', 'test'
    train_bs, test_bs = 256, 512

model_name = 'ResNet18_mulmeta'

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

task_prim_prob = get_primitive_prob('prob.json')
if first14:
    mask_meta = [1.]*14 + [0.]*14
    mask_ft = [0.]*14 + [1.]*14
else:
    mask_meta = mask_ft = [1.] * 28

data_cube = {
        'datahub': DataHub(**data_kw),
        'binary_meta': binary_meta,
        'min_batchsz': 256, 
        'task_prob': normalize(task_prim_prob, mask_meta),
        'task_mask': mask_meta,
        }

lr = 0.5
lr_scheme = {
        'base_lr': lr,
        'lr_policy': 'multistep',
        'gamma': 0.3,
        'stepvalue': (200, 400, ),
        'max_epoch': 600,
        }
lr_inner = .001
lr_scheme_inner = {
        'base_lr': lr_inner,
        'lr_policy': 'multistep',
        'gamma': 0.3,
        'stepvalue': (200, 400, ),
        'max_epoch': 600,
        }
lr_cube = {
        'lr_scheme': lr_scheme,
        'lr_scheme_inner': lr_scheme_inner,
        'k': None,
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
        'pretrain': None,
        'resume': None,
        'optimizer': optim.SGD(model.parameters(), lr=lr),
        'optimizer_inner': optim.Adam(model_inner.parameters(), lr=lr_inner),
            #optim.SGD(model_inner.parameters(), lr=lr_inner, weight_decay=1e-5, momentum=0.9),
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

trainer_meta = Trainer_Meta(model_cube, data_cube, criterion_cube, writer_cube, 
                  lr_cube, snapshot_scheme, device)

train_m, val_m, test_m = trainer_meta.train('f1_macro', verbose_output)
#train_m, val_m, test_m = trainer.test('/home/rongzhao/projects/ml_kaggle_protein/snapshot/ResNet18_multitask_11252204/state_600.pkl')

####################  Finetune Part  ####################
data_cube_ft = {
        'datahub': DataHub(**data_kw),
        'task_mask': mask_ft,
        }

lr_ft = .0001
lr_scheme_ft = {
        'base_lr': lr_ft,
        'lr_policy': 'multistep',
        'gamma': 0.3,
        'stepvalue': (20, 35, 50, ),
        'max_epoch': 60,
        }
lr_cube_ft = {
        'lr_scheme': lr_scheme_ft,
        }
experiment_id_ft = experiment_id + '_ft'
model_cube_ft = {
        'model': model,
        'pretrain': '%s/state_max.pkl' % snapshot_root,
        'resume': None,
#        'optimizer': optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        'optimizer': model_cube['optimizer_inner'],
        }
criterion_cube_ft = {
        'criterion': nn.BCEWithLogitsLoss(weight=None)
        }

snapshot_root_ft = '../snapshot/%s' % (experiment_id_ft)
os.makedirs(snapshot_root, exist_ok=True)
shutil.copy2(P.join('.', this_fname), P.join(snapshot_root, this_fname))

snapshot_scheme_ft = {
        'root': snapshot_root,
        'display_interval': 10,
        'val_interval': 10,
        'snapshot_interval': 999999,
        }

writer_ft = SummaryWriter(log_dir='../tboard/%s' % (experiment_id_ft))
writer_cube_ft = {
        'writer': writer_ft,
        }

trainer = Trainer(model_cube_ft, data_cube_ft, criterion_cube_ft, writer_cube_ft, 
                  lr_cube_ft, snapshot_scheme_ft, device)

train_m_ft, val_m_ft, test_m_ft = trainer.train('f1_macro', verbose_output)



