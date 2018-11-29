#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:10:36 2018

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
from trainer import Metric, write_metric, close_file

class Trainer_Meta(object):
    '''A functional class facilitating and supporting all procedures in training phase'''
    def __init__(self, model_cube, data_cube, criterion_cube, writer_cube, 
                 lr_cube, snapshot_scheme, device, wrap_test=False, task_weight=None):
        self.model, self.model_inner, self.optimizer, self.optimizer_inner, self.start_epoch = \
            self.init_model(model_cube) # Initialize model
        self.root = snapshot_scheme['root']
        self.device = device
        self.parse_dataloader(data_cube)
        self.parse_criterion(criterion_cube)
        self.parse_writer(writer_cube)
        self.lr_scheme = lr_cube['lr_scheme']
        self.lr_scheme_inner = lr_cube['lr_scheme_inner']
        self.snapshot_scheme = snapshot_scheme
        self.max_epoch = self.lr_scheme['max_epoch'] if not wrap_test else 0
        self.task_weight = task_weight
        
        self.lr_inner = self.lr_scheme_inner['base_lr']
        self.k = lr_cube['k']
        
        if not wrap_test:
            with open( P.join(self.root, 'description.txt'), 'w' ) as f:
                f.write(str(lr_cube) + '\n' + str(snapshot_scheme) + '\n' + str(self.model))
        
        self.model.to(self.device)
        self.model.train()
        images, _ = next(iter(self.trainloader))
        l = self.model(images[:1].to(device)).sum()
        l.backward()
        self.model.zero_grad()
        self.model_inner.to(self.device)
        self.model_inner.train()
        
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
        self.model_inner.copy_state(self.model)
        for epoch in range(self.start_epoch, self.max_epoch+1):
            # Adjust learning rate
            adjust_opt(self.optimizer, epoch-1, **self.lr_scheme)
            adjust_opt(self.optimizer_inner, epoch-1, **self.lr_scheme_inner)
            loss = self.train_epoch(verbose)
            if self.lossF:
                self.lossF.write('%d,%.6f\n' % (epoch, loss))
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
                train_metric, val_metric = self.validate_online(self.model, epoch)
                train_metric.print()
                val_metric.print()
                
                train_metric_in, val_metric_in = self.validate_online(self.model_inner, epoch)
                train_metric_in.print('[inner]')
                val_metric_in.print('[inner]')
                write_metric(epoch, train_metric_in, self.trainF)
                write_metric(epoch, val_metric_in, self.valF)
                if max_metric <= val_metric_in.dict[metricOI] and epoch > 10:
                    max_metric = val_metric_in.dict[metricOI]
                    self._snapshot(epoch, 'max')
                    self._snapshot(epoch, 'max_inner', is_inner=True)
                
            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                self.writer.add_scalar('Loss', loss, epoch)
                self.tb_write_scalar(train_metric_in, epoch)
                self.tb_write_scalar(val_metric_in, epoch)
                    
        train_metric, val_metric, test_metric = self.validate_final(self.model)
        train_metric.print()
        val_metric.print()
        test_metric.print()
        
        train_metric_in, val_metric_in, test_metric_in = self.validate_final(self.model_inner)
        train_metric_in.print('[inner]')
        val_metric_in.print('[inner]')
        test_metric_in.print('[inner]')
        self._snapshot(epoch, str(epoch))
        self._snapshot(epoch, str(epoch)+'_inner', is_inner=True)
        close_file(self.trainF)
        close_file(self.valF)
        close_file(self.lossF)
        return train_metric, val_metric, test_metric
        
    def train_epoch(self, verbose=False):
        '''Train the model for one epoch, return the average loss'''
        loss_buf = []
        self.model_inner.copy_state(self.model)
        if self.binary_meta:
            trainloader, cls_id, cls_id_v = self.datahub.taskloader(self.min_batchsz, self.task_prob)
            cls_id_v = torch.FloatTensor(cls_id_v).to(self.device)
            criterion = nn.BCEWithLogitsLoss(cls_id_v)
        else:
            trainloader = self.trainloader
            criterion = nn.BCEWithLogitsLoss(self.task_mask_cuda)
        for i, (images, labels) in enumerate(trainloader, 1):
            # images (N, C, H, W) => (N, 28), labels (N, 28)
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer_inner.zero_grad()
            out = self.model_inner(images)
            loss = criterion(out, labels)
            loss.backward()
            self.optimizer_inner.step()
            # Accumulate gradient in outer model
            self.model.accum_grad(self.model_inner, len(trainloader), self.lr_inner)
#            if i % self.k == 0:
#                self.optimizer.step()
#                self.optimizer.zero_grad()
#                if len(self.trainloader) - i >= self.k:
#                    self.model_inner.copy_state(self.model)
#                else: 
#                    break
            
            loss_buf.append(loss.detach().cpu().numpy())
            if verbose:
                print('The %d-th batch finished: loss = %.3f' % (i, loss_buf[-1]))
        # Update outer model once per epoch
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.array(loss_buf).mean()
    
    def evaluate(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            metric_dict = eval_kernel(model, dataloader, self.device, self.task_mask)
        model.train()
        return metric_dict
    
    def test(self, model, pretrain):
        '''Cordinate the testing of the model after training'''
#        save_dir = P.join(self.root, foldername)
#        pretrain = P.join(self.root, 'state_%s.pkl'%state_suffix)
        self._load_pretrain(pretrain)
        train_metric, val_metric, test_metric = self.validate_final(model)
        train_metric.print()
        val_metric.print()
        test_metric.print()
        return train_metric, val_metric, test_metric
        
    def validate_final(self, model):
        '''Validate the model after training finished, detailed metrics would be recorded'''
        train_metric = Metric(self.evaluate(model, self.trainseqloader), 'Train')
        val_metric = Metric(self.evaluate(model, self.valloader), 'Val')
        test_metric = Metric(self.evaluate(model, self.testloader), 'Test')
        
        return train_metric, val_metric, test_metric
                
    def validate_online(self, model, epoch):
        '''Validate the model during training, record a minimal number of metrics'''
        train_metric = Metric(self.evaluate(model, self.trainseqloader), 'Train')
        val_metric = Metric(self.evaluate(model, self.valloader), 'Val')
        
        return train_metric, val_metric
        
    @staticmethod
    def init_model(model_cube):
        '''Initialize the model, optimizer and related variables according to model_cube'''
        model = model_cube['model']
        model_inner = model_cube['model_inner']
        optimizer = model_cube['optimizer']
        optimizer_inner = model_cube['optimizer_inner']
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
        model_inner.copy_state(model)
        return model, model_inner, optimizer, optimizer_inner, start_epoch
    
    def parse_dataloader(self, data_cube):
        self.datahub = datahub = data_cube['datahub']
        self.trainloader = datahub.trainloader()
        self.valloader= datahub.valloader()
        self.testloader = datahub.testloader()
        self.trainseqloader = datahub.trainseqloader()
        self.min_batchsz = data_cube['min_batchsz']
        self.task_prob = data_cube['task_prob']
        self.task_mask = data_cube['task_mask']
        self.binary_meta = data_cube['binary_meta']
        
        self.task_mask_cuda = torch.FloatTensor(self.task_mask).to(self.device)
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
        
    def _snapshot(self, epoch, name=None, is_inner=False):
        '''Take snapshot of the model, save to root dir'''
#        self.model.to(torch.device('cpu'))
        if is_inner:
            model, optimizer = self.model_inner, self.optimizer_inner
        else:
            model, optimizer = self.model, self.optimizer
        state_dict = {'epoch': epoch,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if name is None:
            filename = '%s/state_%04d.pkl' % (self.root, epoch)
        else:
            filename = '%s/state_%s.pkl' % (self.root, name)
        print('%s Snapshotting to %s' % (timestr(), filename))
        torch.save(state_dict, filename)
        
    def _optim_device(self, device):
        for k, v in self.optimizer.state.items():
            for kk, vv in v.items():
                v[kk] = vv.to(device)