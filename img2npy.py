import csv
from PIL import Image
import numpy as np
import argparse
import os.path as P
import os

def process_img(source, target, all_csv):
    assert P.isdir(source), f'{source} is not a valid directory'
    os.makedirs(target, exist_ok=True)
    with open(all_csv, 'r') as csv_file:
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
                    im = np.array(im, dtype=np.uint8)
                    im = np.expand_dims(im, axis=2)
                    sample.append(im)
                im = np.concatenate(sample, axis=2)
                np.save(f'{target}/{img_id}.npy', im)
                line_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='source image directory', default='train', type=str)
    parser.add_argument('--target', help='target npy directory', default='npy', type=str)
    parser.add_argument('--csv', help='csv file for all idx', default='all.csv', type=str)
    args = parser.parse_args()
    process_img(args.source, args.target, args.csv)
