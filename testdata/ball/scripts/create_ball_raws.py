#
# Copyright (c) 2016,2018-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
import os

from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Batch convert images",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, required=True)
    parser.add_argument('-s','--src',type=str, required=True)
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dest = os.path.abspath(args.dest)
    os.mkdir(dest)
    for root,dirs,files in os.walk(src):
    	for imgs in files:
            img_path = os.path.join(root, imgs)
            if('.png' in img_path):
            	print(img_path)
            	img = Image.open(img_path)
            	img_raw = np.array(img)
            	img_raw = img_raw.astype(np.float32)
            	img_raw = img_raw[..., ::-1]
            	filename, ext = os.path.splitext(imgs)
            	dest_path = os.path.join(dest, filename + ".raw")
            	print("save to %s" % dest_path)
            	img_raw.tofile(dest_path)

if __name__ == '__main__':
    exit(main())