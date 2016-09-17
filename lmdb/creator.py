import numpy as np
import lmdb
import caffe
from PIL import Image

import fnmatch
import os

ROOT_DIR_NAME='/home/prabindh/work-2016/DATASETS/CS-1.6-Dataset/'
matches = []
for root, dirnames, filenames in os.walk(ROOT_DIR_NAME):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        matches.append(os.path.join(root, filename))


N = len(matches)
IMG_WIDTH=1920 #TODO
IMG_HEIGHT=1080
map_size = IMG_WIDTH * IMG_HEIGHT * 3 * N * 10

env = lmdb.open('mylmdb1', map_size=map_size)

with env.begin(write=True) as txn:
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
	img=Image.open(matches[i])
	X = np.array(img)
	y = 1 #TODO - how to provide label names for this ? separate file ? or per folder parsing then use folder name for label ?
        datum.channels = X.shape[2]
        datum.height = X.shape[0]
        datum.width = X.shape[1]
        datum.data = X.tobytes()
        datum.label = y
        str_id = '{:08}'.format(i)
	txn.put(str_id, datum.SerializeToString())
