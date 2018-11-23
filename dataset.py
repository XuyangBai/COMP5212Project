import torch
import torch.utils.data as data
import csv
from PIL import Image
import numpy as np


class HPA(data.Dataset):
    def __init__(self, split='train', transform=None, target_transform=None):
        self.dataset = 'HPA'
        # self.root = 'HPA/train' if split == 'train' else 'HPA/test'
        self.root = 'train' if split == 'train' else 'test'
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.data = {}  # key is img_id, value is multi-label
        # parse csv
        with open('train.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    temp = [int(n) for n in row[1].split(' ')]
                    labels = np.zeros([1, 28])
                    labels[0, temp] = 1
                    self.data[row[0]] = labels.astype(np.long)
                    line_count += 1
            print(f'Processed {line_count} lines.')
        self.img_ids = list(self.data.keys())

    def load_image(self, img_id):
        colors = ['green', 'blue', 'red', 'yellow']
        sample = []
        for color in colors:
            im = Image.open(f'{self.root}/{img_id}_{color}.png')
            im = np.array(im, dtype=np.float32)
            im = np.expand_dims(im, axis=2)
            sample.append(im)
        img = np.concatenate(sample, axis=2)
        return img

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        if self.transform is not None:  # TODO: transform??
            im = self.transform(im)
        label = self.data[img_id]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return im, label

    def __len__(self):
        return len(self.img_ids)
