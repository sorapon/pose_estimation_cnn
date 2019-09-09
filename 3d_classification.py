#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import h5py
from chainer import serializers
import time
import random
from tqdm import tqdm

# multiprocess
from multiprocessing import Process, Value, Array
from multiprocessing import Pool

## 3d show
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import network_euler_class as network

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='epoch size')
parser.add_argument('--batchsize', '-b', default=30, type=int,
                    help='batchsize')
parser.add_argument('--division', '-d', default=50, type=int,
                    help='the number of division for each axis')
parser.add_argument('--num_dataset', '-n', default=50000, type=int, #default=-1
                    help='the number of division for each axis')

args = parser.parse_args()
#xp = cuda.cupy if args.gpu >= 0 else np
gpu = args.gpu
div = args.division
n_data = args.num_dataset
max_epoch = args.epoch
batchsize = args.batchsize

t0 = time.time()

train_num = round(n_data*0.95)
test_num = n_data - train_num

channel = 1
axis_x = div
axis_y = div
axis_z = div
output_pos = 3
output_ori = 3
class_num = 2
output_num = output_pos + output_ori + class_num

x = np.zeros((n_data, channel, axis_z, axis_y, axis_x), dtype='float32')
y = np.zeros((n_data, output_num), dtype='float32')

print("load dataset")
infh = h5py.File('./datasets/paper_hv6_58000.hdf5', 'r')
infh.keys()
infh2 = h5py.File('./datasets/hv7_52000.hdf5', 'r')
infh2.keys()

for n in tqdm(range(0,n_data)):
    if(n < n_data/2):
        voxel_data = infh["data_"+str(n+1)]['voxel'].value
        tf_data = infh["data_"+str(n+1)]['pose'].value
        tf_data[6] = int(1)
        tf_data = np.append(tf_data, int(0))
        print("tf_data : {}".format(tf_data))
    else:
        voxel_data = infh2["data_"+str(n+1)]['voxel'].value
        tf_data = infh2["data_"+str(n+1)]['pose'].value
        tf_data[6] = int(0)
        tf_data = np.append(tf_data, int(1))

    x[n,channel-1] = voxel_data.reshape(axis_x, axis_y, axis_z)
    tf_data[3:6] = tf_data[3:6]/3.14
    y[n] = tf_data

print("tf_data : {}".format(y))
#### visualize voxel data
##    point_x = []
##    point_y = []
##    point_z = []
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##    ax.set_xlabel("x")
##    ax.set_ylabel("y")
##    ax.set_zlabel("z")
##
##    for m in range(channel):
##        for i in range(axis_z):
##            for j in range(axis_y):
##                for k in range(axis_x):
##                    if(voxel_data[(div*div*i) + (div*j) + (k)] == 1):
##                        x[n, m, i, j, k] = 1
##                        point_x.append(k)
##                        point_y.append(j)
##                        point_z.append(i)
##
##    ax.scatter(point_x, point_y, point_z)
##    plt.show()

##for a in range(4):
##p = Process(target=make_voxel, args=(infh,))
##p.start()
##p.join()

print("finish loading datasets")

t1 = time.time()
elapsed_time1 = t1-t0
print("データ読み込み時間：{}".format(elapsed_time1))

p = list(zip(x, y))
random.shuffle(p)
x, y = zip(*p)

from chainer import iterators, optimizers, serializers, report, training
from chainer.datasets import TupleDataset
from chainer.training import extensions

nn = network.CNN()
##print(x[0:train_num])

train = TupleDataset(x[:train_num], y[:train_num])
val = TupleDataset(x[train_num:],y[train_num:])

train_iter = iterators.SerialIterator(train, batchsize)
val_iter = iterators.SerialIterator(val, batchsize, False, False)

optimizer = optimizers.Adam()
optimizer.setup(nn)

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (max_epoch, 'epoch'))

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(val_iter, nn, device=gpu))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.dump_graph('main/loss'))

##trigger = training.triggers.MaxValueTrigger('validation/main/loss', trigger=(1, 'epoch'))
##trainer.extend(extensions.snapshot_object(nn, filename='result/new_best.model'), trigger=trigger)
trainer.run()

if args.gpu >= 0:
    nn.to_cpu()

t2 = time.time()
elapsed_time2 = t2-t1
print("データ読み込み時間：{}".format(elapsed_time1))
print("学習時間：{}".format(elapsed_time2))

serializers.save_npz('result/new.model', nn)
##plt.plot(x,y)
##plt.plot(x,y_pred.data)
##plt.show()
