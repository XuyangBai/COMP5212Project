#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:00:14 2018

@author: rongzhao
"""

import torch
import torch.nn as nn
import numpy as np
import os.path as P
from model import resnet18_protein
from utils import timestr, adjust_opt
from evaluator import evaluate as eval_kernel
import collections

class Dummy_Dataloader(object):
    '''Create a dummy dataloader to test trainer'''
    def __init__(self, length, bsize=32, channels=4, H=512, W=512, nClass=28):
        self.data = torch.ones((bsize, channels, H, W))
        self.data.normal_()
        self.label = torch.ones((bsize, nClass))
        self.label.bernoulli_()
        self.len = length
        self.outstanding_batches = self.len
        
    def __iter__(self):
        self.outstanding_batches = self.len
        return self
    
    def __next__(self):
        if self.outstanding_batches <= 0:
            raise StopIteration
        self.outstanding_batches -= 1
        return self.data, self.label
    
class Metric(object):
    '''Metric class manages the record and log of classification metrics'''
    def __init__(self, metric_dict, split):
        self.dict = metric_dict
        self.split = split
        
    def print(self, prewords=''):
        pstr = prewords
        for k, v in self.dict.items():
            if isinstance(v, collections.Iterable):
                continue
            pstr += '%s = %.3f ' % (k, v)
        print(self.split + ': ' + pstr)
            
def close_file(F):
    if F:
        F.close()

def write_metric(metric, writeF):
    if writeF:
        d = metric.dict
        writeF.write('%.4f,%.4f,%.4f,%.4f\n' % (d['acc'], d['f1_macro'], d['prec'], d['recl']))
        writeF.flush()

class Trainer(object):
    '''A functional class facilitating and supporting all procedures in training phase'''
    def __init__(self, model_cube, data_cube, criterion_cube, writer_cube, 
                 lr_cube, snapshot_scheme, device, wrap_test=False, task_weight=None):
        self.model, self.optimizer, self.start_epoch = \
            self.init_model(model_cube) # Initialize model
#        if model_cube['resume']:
#            self._optim_device(device)
        self.root = snapshot_scheme['root']
        self.parse_dataloader(data_cube)
        self.parse_criterion(criterion_cube)
        self.parse_writer(writer_cube)
        self.lr_scheme = lr_cube['lr_scheme']
        self.snapshot_scheme = snapshot_scheme
        self.max_epoch = self.lr_scheme['max_epoch'] if not wrap_test else 0
        self.device = device
        self.task_weight = task_weight
        
        if not wrap_test:
            with open( P.join(self.root, 'description.txt'), 'w' ) as f:
                f.write(str(self.lr_scheme) + '\n' + str(snapshot_scheme) + '\n' + str(self.model))
        self.model.to(self.device)
        self.model.train()
        
    def tb_write_scalar(self, metric, epoch):        
        for k, v in metric.dict.items():
            if isinstance(v, collections.Iterable):
                continue
            self.writer.add_scalar('%s/%s' % (metric.split, k), v, epoch)
        
    def train(self, metricOI='f1_macro', verbose=False):
        '''Cordinate the whole training phase, mainly recording of losses and metrics, 
        lr and loss weight decay, snapshotting, etc.'''
        loss_all = []
        max_metric = 0
        print(timestr(), 'Optimization Begin')
        for epoch in range(self.start_epoch, self.max_epoch+1):
            # Adjust learning rate
            adjust_opt(self.optimizer, epoch-1, **self.lr_scheme)
            loss = self.train_epoch(verbose)
            if self.lossF:
                self.lossF.write('%.6f\n' % loss)
                self.lossF.flush()
            loss_all.append(loss)
            
            if epoch % self.snapshot_scheme['display_interval'] == 0 or epoch == self.start_epoch:
                N = self.snapshot_scheme['display_interval']
                loss_avg = np.array(loss_all[-N:]).mean()
                first_epoch = epoch if epoch == self.start_epoch else epoch+1-N
                print('%s Epoch %d ~ %d: loss = %.7f, current lr = %.7e' %
                      (timestr(), first_epoch, epoch, loss_avg, self._get_lr()))
            
            if epoch % self.snapshot_scheme['snapshot_interval'] == 0 or epoch == self.start_epoch:
                self._snapshot(epoch)
            
            if epoch % self.snapshot_scheme['val_interval'] == 0 or epoch == self.start_epoch:
                train_metric, val_metric = self.validate_online(epoch)
                train_metric.print()
                val_metric.print()
                write_metric(train_metric, self.trainF)
                write_metric(val_metric, self.valF)
                if max_metric <= val_metric.dict[metricOI] and epoch > 10:
                    max_metric = val_metric.dict[metricOI]
                    self._snapshot(epoch, 'max')
            
            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                self.writer.add_scalar('Loss', loss, epoch)
                self.tb_write_scalar(train_metric, epoch)
                self.tb_write_scalar(val_metric, epoch)
                    
        train_metric, val_metric, test_metric = self.validate_final()
        train_metric.print()
        val_metric.print()
        test_metric.print()
        self._snapshot(epoch, str(epoch))
        close_file(self.trainF)
        close_file(self.valF)
        close_file(self.lossF)
        return train_metric, val_metric, test_metric
        
    def train_epoch(self, verbose=False):
        '''Train the model for one epoch, return the average loss'''
        loss_buf = []
        for i, (images, labels) in enumerate(self.trainloader):
            # images (N, C, H, W) => (N, 28), labels (N, 28)
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(images)
            criterion = nn.BCEWithLogitsLoss(self.task_weight)
            loss = criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())
            if verbose:
                print('The %d-th batch finished: loss = %.3f' % (i, loss_buf[-1]))
        
        return np.array(loss_buf).mean()
    
    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            metric_dict = eval_kernel(self.model, dataloader, self.device, self.task_mask)
        self.model.train()
        return metric_dict
    
    def test(self, pretrain):
        '''Cordinate the testing of the model after training'''
#        save_dir = P.join(self.root, foldername)
#        pretrain = P.join(self.root, 'state_%s.pkl'%state_suffix)
        self._load_pretrain(pretrain)
        train_metric, val_metric, test_metric = self.validate_final()
        train_metric.print()
        val_metric.print()
        test_metric.print()
        return train_metric, val_metric, test_metric
        
    def validate_final(self):
        '''Validate the model after training finished, detailed metrics would be recorded'''
        train_metric = Metric(self.evaluate(self.trainseqloader), 'Train')
        val_metric = Metric(self.evaluate(self.valloader), 'Val')
        test_metric = Metric(self.evaluate(self.testloader), 'Test')
        
        return train_metric, val_metric, test_metric
                
    def validate_online(self, epoch):
        '''Validate the model during training, record a minimal number of metrics'''
        train_metric = Metric(self.evaluate(self.trainseqloader), 'Train')
        val_metric = Metric(self.evaluate(self.valloader), 'Val')
        
        return train_metric, val_metric
        
    @staticmethod
    def init_model(model_cube):
        '''Initialize the model, optimizer and related variables according to model_cube'''
        model = model_cube['model']
        optimizer = model_cube['optimizer']
        pretrain = model_cube['pretrain']
        resume = model_cube['resume']
        start_epoch = 1
        if resume:
            if P.isfile(resume):
                model.cpu()
                state = torch.load(resume)
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                start_epoch = state['epoch'] + 1
            else:
                raise RuntimeError('No checkpoint found at %s' % pretrain)
        elif pretrain:
            if P.isfile(pretrain):
                model.cpu()
                state = torch.load(pretrain)
                model.load_state_dict(state['state_dict'])
            else:
                raise RuntimeError('No checkpoint found at %s' % pretrain)
        
        return model, optimizer, start_epoch
    
    def parse_dataloader(self, data_cube):
        self.datahub = datahub = data_cube['datahub']
        self.task_mask = data_cube['task_mask']
        self.trainloader = datahub.trainloader()
        self.valloader= datahub.valloader()
        self.testloader = datahub.testloader()
        self.trainseqloader = datahub.trainseqloader()
#        self.min_batchsz = data_cube['min_batchsz']
#        self.task_prob = data_cube['task_prob']
#        self.val_sn = data_cube.val_sn()
#        self.test_sn = data_cube.test_sn()
#        self.train_sn = data_cube.train_sn()
        
    def parse_criterion(self, criterion_cube):
        if criterion_cube is None:
            return
        self.criterion = criterion_cube['criterion']
        
    def parse_writer(self, writer_cube):
        if writer_cube is None:
            return
        self.writer = writer_cube['writer']
        trainFn, valFn, lossFn = writer_cube.get('trainF', None), \
        writer_cube.get('valF', None), writer_cube.get('lossF', None)
        
        self.trainF = self.valF = self.lossF = None
        if trainFn:
            self.trainF = open(P.join(self.root, trainFn), 'w')
        if valFn:
            self.valF = open(P.join(self.root, valFn), 'w')
        if lossFn:
            self.lossF = open(P.join(self.root, lossFn), 'w')
    
    def _load_pretrain(self, pretrain):
#        self.model.cpu()
        state = torch.load(pretrain)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
    
    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
        
    def _snapshot(self, epoch, name=None):
        '''Take snapshot of the model, save to root dir'''
#        self.model.to(torch.device('cpu'))
        state_dict = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if name is None:
            filename = '%s/state_%04d.pkl' % (self.root, epoch)
        else:
            filename = '%s/state_%s.pkl' % (self.root, name)
        print('%s Snapshotting to %s' % (timestr(), filename))
        torch.save(state_dict, filename)
#        self.model.to(self.device)
        
    def _optim_device(self, device):
        for k, v in self.optimizer.state.items():
            for kk, vv in v.items():
                v[kk] = vv.to(device)


class TrainerX(object):
    '''Trainer class maintainance all procedures of the training phase'''
    def __init__(self, model, dataloader, optimizer, device='cuda:0', writer=None, 
                 task_weight=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.task_weight = task_weight
        
        self.model.to(device)
        if self.task_weight != None:
            self.task_weight = torch.Tensor(self.task_weight).to(device)
            
    def train_epoch(self):
        '''Train the model for one epoch, return the average loss'''
        loss_buf = []
        for images, labels in iter(self.dataloader):
            # images (N, C, H, W) => (N, 28), labels (N, 28)
            images, labels = images.to(self.device), labels.to(self.device)
            labels.squeeze_(dim=1)
            self.optimizer.zero_grad()
            out = self.model(images)
            criterion = nn.BCEWithLogitsLoss(self.task_weight)
            loss = criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())
        
        return np.array(loss_buf).mean()
        
    def train():
        pass
    
if __name__ == '__main__':
    model = resnet18_protein()
    dl = Dummy_Dataloader(10)
    optim = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
    trainer = TrainerX(model, dl, optim)
    loss = trainer.train_epoch()
    
    
    
    
    