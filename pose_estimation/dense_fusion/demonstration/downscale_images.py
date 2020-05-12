import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Downscale images in folder')
parser.add_argument('--path', type=str, default='pose_estimation/dense_fusion/demonstration/data', help='path/to/folder')
parser.add_argument('--type', type=str, default='png', help='image type (default: png)')
args = parser.parse_args()


def main():
    root = args.path
    img_type = f'.{args.type}'

    bar = tqdm(os.listdir(root))
    for file in bar:
        if file.endswith(img_type):
            path = f'{root}/{file}'
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img)


if __name__ == '__main__':
    main()
