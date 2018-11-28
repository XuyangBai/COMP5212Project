#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:29:23 2018

@author: rongzhao
"""

import time
import numpy as np
import os
import os.path as osp
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
# from torchmed import transforms, datasets
import json


def get_primitive_prob():
    with open('prob.json', 'r') as f:
        class_to_prob = json.load(f)
    prob = [v for k, v in class_to_prob.items()]
    prob[-1] = 1 - sum(prob[0:-1])
    return np.array(prob, dtype=np.float32)


def normalize(nums):
    nums = nums / np.sum(nums)
    return nums


def timestr(form=None):
    if form is None:
        return time.strftime("<%Y-%m-%d %H:%M:%S>", time.localtime())
    if form == 'mdhm':
        return time.strftime('%m%d%H%M', time.localtime())


def adjust_lr(i, epoch, max_epoch, **kw):
    lr_policy = kw.get('lr_policy')
    base_lr = kw.get('base_lr')
    lr_policy = lr_policy[i] if isinstance(lr_policy, (tuple, list)) else lr_policy
    base_lr = base_lr[i] if isinstance(base_lr, (tuple, list)) else base_lr
    if lr_policy == 'fixed':
        return base_lr
    elif lr_policy == 'step':
        gamma = kw.get('gamma')
        stepsize = kw.get('stepsize')
        return base_lr * gamma ** math.floor(epoch / stepsize)
    elif lr_policy == 'exp':
        gamma = kw.get('gamma')
        return base_lr * gamma ** epoch
    elif lr_policy == 'inv':
        gamma = kw.get('gamma')
        power = kw.get('power')
        return base_lr * (1 + gamma * epoch) ** (-power)
    elif lr_policy == 'multistep':
        gamma = kw.get('gamma')
        stepvalue = kw.get('stepvalue')
        lr = base_lr
        for value in stepvalue:
            if epoch >= value:
                lr *= gamma
            else:
                break
        return lr
    elif lr_policy == 'poly':
        power = kw.get('power')
        return base_lr * (1 - epoch / max_epoch) ** power
    elif lr_policy == 'sigmoid':
        gamma = kw.get('gamma')
        stepsize = kw.get('stepsize')
        return base_lr * (1 / (1 + math.exp(-gamma * (epoch - stepsize))))
    else:
        raise RuntimeError('Unknown lr_policy: %s' % lr_policy)


def adjust_opt(optimizer, epoch, max_epoch, **kw):
    for i, param in enumerate(optimizer.param_groups):
        param['lr'] = adjust_lr(i, epoch, max_epoch, **kw)


class DataCubeNaive(object):
    def __init__(self, train_loader, val_loader, test_loader, trainseq_loader):
        self._trainloader = train_loader
        self._valloader = val_loader
        self._testloader = test_loader
        self._trainseqloader = trainseq_loader

    def trainloader(self):
        return self._trainloader

    def valloader(self):
        return self._valloader

    def testloader(self):
        return self._testloader

    def trainseqloader(self):
        return self._trainseqloader


class DataHub_ProteinCls(object):
    def __init__(self, root, train_split, val_split, test_split, datapath, train_batchsize,
                 test_batchsize, modalities, std, mean,
                 rand_flip=None, crop_type=None, crop_size_img=None,
                 balance_rate=0.5, train_pad_size=None, mod_drop_rate=0,
                 train_drop_last=False,
                 DataSet=datasets.MMDataset,
                 label_loader_path=None, weighted_sample_rate=None, rand_rot90=False,
                 num_workers=1, mem_shape=None, random_black_patch_size=None):
        self.root = root
        self.std = std
        self.mean = mean
        self.num_workers = num_workers
        self.mem_shape = mem_shape

        TestDataSet = DataSet
        datapath_test = datapath
        modal_test = modalities

        if train_split:
            with open(osp.join(root, train_split), 'r') as f:
                self._train_sn = f.read().splitlines()
        if val_split:
            with open(osp.join(root, val_split), 'r') as f:
                self._val_sn = f.read().splitlines()
        if test_split:
            with open(osp.join(root, test_split), 'r') as f:
                self._test_sn = f.read().splitlines()

        if osp.exists(osp.join(root, datapath, 'meanstd.txt')):
            with open(osp.join(root, datapath, 'meanstd.txt'), 'r') as f:
                lines = f.read().splitlines()
            self.mean = [float(x) for x in lines[0].split()[1:]]
            self.std = [float(x) for x in lines[1].split()[1:]]
            print('import mean and std value from file \'meanstd.txt\'')
            print('mean = %s, std = %s' % (str(self.mean), str(self.std)))

        self.basic_transform_ops = [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]

        train_transform = \
            self._make_train_transform(crop_type, crop_size_img,
                                       rand_flip, mod_drop_rate, balance_rate,
                                       train_pad_size, rand_rot90, random_black_patch_size)
        test_transform = \
            self._make_test_transform()

        self._trainloader = \
            self._make_dataloader(DataSet, train_split, datapath, train_batchsize, train_transform,
                                  modalities, shuffle=True, drop_last=train_drop_last,
                                  weighted_sample_rate=weighted_sample_rate)

        self._trainseqloader = \
            self._make_dataloader(TestDataSet, train_split, datapath_test, test_batchsize,
                                  test_transform, modal_test, shuffle=False)

        self._valloader = \
            self._make_dataloader(TestDataSet, val_split, datapath_test, test_batchsize,
                                  test_transform, modal_test, shuffle=False)

        self._testloader = \
            self._make_dataloader(TestDataSet, test_split, datapath_test, test_batchsize,
                                  test_transform, modal_test, shuffle=False)

        self._trainseqloader_label = \
            self._make_labelloader(train_split, label_loader_path, test_batchsize)

        self._valloader_label = \
            self._make_labelloader(val_split, label_loader_path, test_batchsize)

        self._testloader_label = \
            self._make_labelloader(test_split, label_loader_path, test_batchsize)

    def _make_dataloader(self, DataSet, split, datapath, batch_size, transform, modalities,
                         shuffle=False, drop_last=False, weighted_sample_rate=None):
        if split is None:
            return None
        if (self.mem_shape is not None) and DataSet == datasets.MMDataset_memmap:
            data_set = DataSet(self.root, split, datapath, modalities, transform, self.mem_shape)
        else:
            data_set = DataSet(self.root, split, datapath, modalities, transform)
        sampler = None
        if weighted_sample_rate is not None:
            weights = np.where(data_set.get_mask() == 1, weighted_sample_rate[1],
                               weighted_sample_rate[0]).tolist()
            #            weights = torch.from_numpy(weights).float()
            sampler = WeightedRandomSampler(weights, len(data_set), replacement=True)
            shuffle = False
        data_loader = DataLoader(data_set, batch_size=batch_size, sampler=sampler,
                                 shuffle=shuffle, num_workers=self.num_workers, pin_memory=False,
                                 drop_last=drop_last)
        return data_loader

    def _make_labelloader(self, split, datapath, batch_size):
        if datapath is None:
            return None
        data_set = datasets.VanillaDataset(self.root, split, datapath)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=False, drop_last=False)
        return data_loader

    def _make_train_transform(self, crop_type, crop_size_img,
                              rand_flip, mod_drop_rate, balance_rate, pad_size,
                              rand_rot90, random_black_patch_size):
        train_transform_ops = self.basic_transform_ops.copy()

        train_transform_ops += [transforms.RandomBlack(random_black_patch_size),
                                transforms.RandomDropout(mod_drop_rate),
                                transforms.RandomFlip(rand_flip)]
        if pad_size is not None:
            train_transform_ops.append(transforms.Pad(pad_size, 0))

        if rand_rot90:
            train_transform_ops.append(transforms.RandomRotate2d())

        if crop_type == 'random':
            train_transform_ops.append(transforms.RandomCrop(crop_size_img))
        elif crop_type == 'balance':
            train_transform_ops.append(transforms.BalanceCrop(balance_rate, crop_size_img))
        elif crop_type == 'center':
            train_transform_ops.append(transforms.CenterCrop(crop_size_img))
        elif crop_type is None:
            pass
        else:
            raise RuntimeError('Unknown train crop type.')

        return transforms.Compose(train_transform_ops)

    def _make_test_transform(self):
        test_transform_ops = self.basic_transform_ops.copy()
        return transforms.Compose(test_transform_ops)

    def trainloader(self):
        return self._trainloader

    def valloader(self):
        return self._valloader

    def testloader(self):
        return self._testloader

    def trainseqloader(self):
        return self._trainseqloader

    def valloader_label(self):
        return self._valloader_label

    def testloader_label(self):
        return self._testloader_label

    def trainseqloader_label(self):
        return self._trainseqloader_label

    def train_sn(self):
        return self._train_sn

    def val_sn(self):
        return self._val_sn

    def test_sn(self):
        return self._test_sn
