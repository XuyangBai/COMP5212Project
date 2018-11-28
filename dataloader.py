import numpy as np
import random
import PIL
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, SubsetRandomSampler
import transforms as mytfm
import torchvision.transforms as transforms
import json
from dataset import HPA

# torch.multiprocessing.freeze_support()


def get_sampler(split, inst_num=None, prob=None):
    if prob is None:
        with open('prob.json', 'r') as f:
            class_to_prob = json.load(f)
        prob = [v for k, v in class_to_prob.items()]
        prob[-1] = 1 - sum(prob[0:-1])
    class_id = int(np.random.choice(range(28), p=prob))
    # Sample num_inst instances from selected class & num_inst from other selected class
    with open(f'preprocess/{class_id}.txt', 'r') as f:
        instance_ids = f.readlines()
        instance_ids = [int(id.split(",")[0]) for id in instance_ids]
    if inst_num is None:
        all = np.array(instance_ids)
    else:
        with open(f'{split}.csv', 'r') as f:
            all_instances_id = f.readlines()
        train_ids = instance_ids
        train_ids_other = np.random.choice(np.arange(len(all_instances_id)), inst_num - len(instance_ids))
        all = np.concatenate((train_ids, train_ids_other), axis=0)
    np.random.shuffle(all)
    return SubsetRandomSampler(all), class_id


def get_data_loader_meta_learning(root, batch_size, preprocess, split='train', inst_num=None, sequential=False, num_workers=4,
                                  prob=None):
    dset = HPA(root=root, split='all', transform=preprocess)
    sampler, class_id = get_sampler(split=split, inst_num=inst_num, prob=prob)
    loader = DataLoader(dset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return loader, class_id


class DataHub(object):
    '''A handy wrap of dataloader'''

    def __init__(self, root, train_bs, test_bs, train_sp, val_sp, test_sp, mean, std,
                 train_flip=None, train_crop=None, train_black=None, test_crop=None,
                 num_workers=4):
        self.root = root
        self.train_bs = train_bs
        self.num_workers = num_workers
        self._base_tfm = [transforms.ToTensor(),
                          transforms.Normalize(mean=mean, std=std)]
        train_tfm = self.get_train_tfm(train_flip, train_crop, train_black)
        self.train_tfm = train_tfm
        self._trainloader = get_data_loader(root, train_bs, train_tfm, train_sp,
                                            False, num_workers)

        test_tfm = self.get_test_tfm(test_crop)
        self._valloader = get_data_loader(root, test_bs, test_tfm, val_sp,
                                          True, num_workers)
        self._testloader = get_data_loader(root, test_bs, test_tfm, test_sp,
                                           True, num_workers)
        self._trainseqloader = get_data_loader(root, test_bs, test_tfm, train_sp,
                                               True, num_workers)

    def trainloader(self):
        return self._trainloader

    def valloader(self):
        return self._valloader

    def testloader(self):
        return self._testloader

    def trainseqloader(self):
        return self._trainseqloader

    def taskloader(self, inst_num=None, prob=None):
        return get_data_loader_meta_learning(self.root, self.train_bs, 'train', inst_num, self.train_tfm, False,
                                             self.num_workers, prob)

    def get_train_tfm(self, train_flip, train_crop, train_black):

        tfm_list = self._base_tfm.copy()
        if train_flip is not None:
            tfm_list.append(mytfm.RandomFlip2d_cls(train_flip))
        if train_crop is not None:
            tfm_list.append(mytfm.RandomCrop2d_cls(train_crop))
        if train_black is not None:
            tfm_list.append(mytfm.RandomBlack2d_cls(train_black))

        return transforms.Compose(tfm_list)

    def get_test_tfm(self, test_crop):
        tfm_list = self._base_tfm.copy()
        if test_crop is not None:
            tfm_list.append(mytfm.CenterCrop2d_cls(test_crop))

        return transforms.Compose(tfm_list)


def get_data_loader(root, batch_size, preprocess, split='train', sequential=False,
                    num_workers=4):
    # # G R B Y
    # mean = [13.528, 20.535, 14.249, 21.106]
    # std = [28.700, 38.161, 40.196, 38.172]
    # # mean =  [x / 255.0 for x in [13.528, 20.535, 14.249, 21.106]]
    # # std = [x / 255.0 for x in [28.700, 38.161, 40.196, 38.172]]
    # normalize = transforms.Normalize(
    #     mean=mean,
    #     std=std
    # )
    # preprocess = transforms.Compose([
    #     # transforms.Resize(128, PIL.Image.BILINEAR),
    #     # transforms.RandomCrop(224),
    #     transforms.ToTensor(),
    #     mytfm.RandomFlip2d_cls((1, 1)),
    #     mytfm.RandomCrop2d_cls((384, 384)),
    #     normalize
    # ])
    dset = HPA(root=root, split=split, transform=preprocess)
    if sequential is True:
        sampler = SequentialSampler(dset)
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
