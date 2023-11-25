#
# Copyright (c) 2016,2018-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
from numpy.linalg import inv
import os
import cv2

modelInputWidth = 1600
modelInputHeight = 480
oriWidth = 7600
oriHeight = 2160
scaleW = float(modelInputWidth) / float(oriWidth)
scaleH = float(modelInputHeight) / float(oriHeight)
scale = min(scaleW, scaleH)
i2d = np.array([
    [scale, 0, (-scale * oriWidth + modelInputWidth + scale - 1) * 0.5],
    [0, scale, (-scale * oriHeight + modelInputHeight + scale - 1) * 0.5]
])
d2i = cv2.invertAffineTransform(i2d)

def getTopK(heatmap, width, height, topK):
    thresh = 0.008
    distanceThresh = 25.0
    res = []
    keyPoints = []
    size = width * height
    for i in range(0, size):
        if heatmap[i] > thresh:
            row = i / width
            col = i % width
            keyPoints.append((i, row, col, heatmap[i]))

    if len(keyPoints) == 0:
        index = heatmap.argmax()
        score = heatmap[index]
        res.append((index, score))
        res.append((index, score))
        return res

    keyPoints.sort(key=lambda a: a[3], reverse=True)
    remove_flags = [False] * len(keyPoints)
    output = []
    for idx, a in enumerate(keyPoints):
        if remove_flags[idx] == True:
            continue
        output.append(a)
        if len(output) >= topK:
            break
        for j in range(i + 1, len(keyPoints)):
            if remove_flags[j] == True:
                continue
            b = keyPoints[j]
            dis = (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
            if dis <= distanceThresh:
                remove_flags[j] = True

    res.append((output[0][0], output[0][3]))
    if len(output) < topK:
        res.append((output[0][0], output[0][0]))
    else:
        res.append((output[1][0], output[1][3]))

    return res

def getPointInfo(index, modelOutputWidth, modelOutputHeight):
    hx = index % modelOutputWidth / float(modelOutputHeight)
    hy = index / modelOutputWidth / float(modelOutputHeight)
    hx = hx * modelInputWidth * d2i[0, 0] * d2i[0, 2]
    hy = hy * modelInputHeight * d2i[1, 1] * d2i[1, 2]
    return (hx, hy)

def main():
    parser = argparse.ArgumentParser(description="show result",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--src',type=str, required=True)
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    for root,dirs,files in os.walk(src):
    	for output in files:
            output_path = os.path.join(root, output)
            print("output is %s" % output_path)
            data = np.fromfile(output_path, dtype='float32')
            res = getTopK(data, 400, 120, 2)
            p1 = getPointInfo(res[0][0], 400, 120)
            p2 = getPointInfo(res[1][0], 400, 120)
            pos = [p1[0], p1[1], res[0][1], p2[0], p2[1], res[1][1]]
            print(pos)

if __name__ == '__main__':
    exit(main())