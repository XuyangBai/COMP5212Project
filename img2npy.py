import csv
from PIL import Image
import numpy as np
import argparse


def process_img(source, target):
    with open('git.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                img_id = row[0]
                colors = ['green', 'red', 'blue', 'yellow']
                sample = []
                for color in colors:
                    im = Image.open(f'{source}/{img_id}_{color}.png')
                    im = np.array(im, dtype=np.int8)
                    im = np.expand_dims(im, axis=2)
                    sample.append(im)
                im = np.concatenate(sample, axis=2)
                np.save(f'{target}/{img_id}.npy', im)
                line_count += 1


parser = argparse.ArgumentParser()
parser.add_argument('--source', help='source image dictionary', default='train', type=str)
parser.add_argument('--target', help='target npy dictionary', default='npy', type=str)
args = parser.parse_args()
process_img(args.source, args.target)
