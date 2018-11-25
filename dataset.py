import torch
import torch.utils.data as data
import csv
from PIL import Image
import numpy as np


class HPA(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.dataset = 'HPA'
        self.root = root # 把解压后的train文件夹放在dataset目录下
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.data = {}  # key is img_id, value is multi-label
        # parse csv
        with open(f'{split}.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    temp = [int(n) for n in row[1].split(' ')]
                    labels = np.zeros([28])
                    labels[temp] = 1
                    self.data[row[0]] = labels.astype(np.float32)
                    line_count += 1
            print(f'Processed {line_count} lines.')
        self.img_ids = list(self.data.keys())

    def load_image(self, img_id):
        # colors = ['green', 'red', 'blue', 'yellow']
        # sample = []
        # for color in colors:
        #     im = Image.open(f'{self.root}/train/{img_id}_{color}.png')
        #     im = np.array(im, dtype=np.float32)
        #     im = np.expand_dims(im, axis=2)  # transform will convert HWC in [0,255] to CHW [0,1]
        #     sample.append(im)
        # img = np.concatenate(sample, axis=2)
        img = np.load(f'{self.root}/{img_id}.npy')#.astype('float32')
        return img.astype('float32')

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
#        print('im:' + str(im.dtype) + str(im.shape))
        if self.transform is not None:
            im = self.transform(im)
#        print('im:' + str(im.dtype) + str(im.size()))
        label = self.data[img_id]
#        print('label before tfm: ' + str(label.dtype) + str(label.shape))
        if self.target_transform is not None:
            label = self.target_transform(label)
#        print('label after tfm: ' + str(label.dtype))
        return im, label

    def __len__(self):
        return len(self.img_ids)
