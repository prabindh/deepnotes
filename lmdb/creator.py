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
TARGET_IMG_SIZE=96 #TODO
map_size = TARGET_IMG_SIZE * TARGET_IMG_SIZE * 3 * N * 10
initLabel = 1
y = initLabel

env = lmdb.open('mylmdb_train', map_size=map_size)
with env.begin(write=True) as txn:
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
	img=Image.open(matches[i])
	X = np.array(img.resize((TARGET_IMG_SIZE,TARGET_IMG_SIZE), Image.ANTIALIAS))
	y = (~y) #Random TODO - how to provide label names for this ? separate file ? or per folder parsing then use folder name for label ?
        datum.channels = X.shape[2]
        datum.height = X.shape[0]
        datum.width = X.shape[1]
        datum.data = X.tobytes()
        datum.label = y
        str_id = '{:08}'.format(i)
	txn.put(str_id, datum.SerializeToString())

env = lmdb.open('mylmdb_val', map_size=map_size)
with env.begin(write=True) as txn:
    for i in range(10):
        datum = caffe.proto.caffe_pb2.Datum()
	img=Image.open(matches[i])
	X = np.array(img.resize((TARGET_IMG_SIZE,TARGET_IMG_SIZE), Image.ANTIALIAS))
	y = ~y #TODO - how to provide label names for this ? separate file ? or per folder parsing then use folder name for label ?
        datum.channels = X.shape[2]
        datum.height = X.shape[0]
        datum.width = X.shape[1]
        datum.data = X.tobytes()
        datum.label = y
        str_id = '{:08}'.format(i)
	txn.put(str_id, datum.SerializeToString())
