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
from utils import DataCube
from model import resnet18_protein
from dataloader import get_data_loader


device = torch.device('cuda:0')
timestr = time.strftime('%m%d%H%M')
this_fname = 'train.py'

verbose_output = False
is_temp = True

train_bs, test_bs = 128, 256
data_root = 'npy'
train_loader = get_data_loader(data_root, train_bs, split='train', sequential=False)
val_loader = get_data_loader(data_root, train_bs, split='validation', sequential=True)
test_loader = get_data_loader(data_root, train_bs, split='test', sequential=True)
trainseq_loader = get_data_loader(data_root, train_bs, split='train', sequential=True)

data_cube = DataCube(train_loader, val_loader, test_loader, trainseq_loader)

lr = 0.03
lr_scheme = {
        'base_lr': lr,
        'lr_policy': 'multistep',
#        'lr_policy': 'step', 
        'gamma': 0.3,
        'stepvalue': (30, 60, 90, ),
#        'stepsize': 1000,
        'max_epoch': 100,
        }
#model = M.Toy_alpha(1, 2)
#num_mo=3
#experiment_id = 'Toy_%s' % timestr

model = resnet18_protein()
if is_temp:
    experiment_id = 'ResNet18_multitask_temp' #_%s' % timestr
else:
    experiment_id = 'ResNet18_multitask_%s' % timestr
model_cube = {
        'model': model,
#        'init_func': misc.weights_init,
        'pretrain': None,
        'resume': None,
        'optimizer': optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
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
trainer.train('accuracy', verbose_output)



