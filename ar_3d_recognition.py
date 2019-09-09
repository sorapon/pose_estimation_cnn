import numpy as np

import matplotlib.pyplot as plt
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

import h5py
from chainer import serializers
import cupy

import chainer.computational_graph as c
import network

import time

## 3d show
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int, #default=-1
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--division', '-d', default=50, type=int,
                    help='the number of division for each axis')
parser.add_argument('--num_dataset', '-n', default=5, type=int, #default=-1
                    help='the number of division for each axis')

args = parser.parse_args()
gpu = args.gpu
div = args.division
xp = cupy if gpu >= 0 else np

model_path = "result/"
model_name = "pos_ori.model"

n_data = args.num_dataset
channel = 1
axis_x = div
axis_y = div
axis_z = div
output_pos = 3
output_ori = 4

x = np.zeros((n_data, channel, axis_z, axis_y, axis_x))
y = np.zeros((n_data,7))

##with cupy.cuda.Device(0):
##    x = cupy.array(x)

print("load dataset")
infh = h5py.File('./datasets/cloud_pos_ori_2030.hdf5', 'r')
infh.keys()

print(x.shape)
print(y.shape)
for n in range(0,n_data):
    voxel_data = infh["data_"+str(n+2010)]['voxel'].value
    tf_data = infh["data_"+str(n+2010)]['pos_ori'].value
    y[n] = tf_data

## visualize voxel data
    point_x = []
    point_y = []
    point_z = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    t0 = time.time()
    for m in range(channel):
        for i in range(axis_z):
            for j in range(axis_y):
                for k in range(axis_x):
                    if(voxel_data[(div*div*i) + (div*j) + (k)] == 1):
                        x[n, m, i, j, k] = 1
##                        point_x.append(k)
##                        point_y.append(j)
##                        point_z.append(i)
##
##    ax.scatter(point_x, point_y, point_z)
##    plt.show()


t1 = time.time()
input_cloud = x.astype(xp.float32)
output_label = y.astype(xp.float32)


nn = network.CNN()
serializers.load_npz(model_path+model_name, nn)

##if args.gpu >= 0:
##    chainer.cuda.get_device(args.gpu).use()
##    nn.to_gpu(gpu)

y_pre = nn.predict(input_cloud)
t2 = time.time()
elapsed_time1 = t1-t0
elapsed_time2 = t2-t1
print("voxel化時間：{}".format(elapsed_time1))
print("推定時間：{}".format(elapsed_time2))

print(output_label)

y = y_pre.data
y_pos = y[0,0:3]
y_ori = y[0,3:7]

y_pos = 0.1*y_pos
nrm = y_ori[0]*y_ori[0] + y_ori[1]*y_ori[1] + y_ori[2]*y_ori[2] + y_ori[3]*y_ori[3]
y_ori_x = np.sign(y_ori[0])*np.sqrt((y_ori[0]*y_ori[0]/float(nrm)))
y_ori_y = np.sign(y_ori[1])*np.sqrt((y_ori[1]*y_ori[1]/float(nrm)))
y_ori_z = np.sign(y_ori[2])*np.sqrt((y_ori[2]*y_ori[2]/float(nrm)))
y_ori_w = np.sign(y_ori[3])*np.sqrt((y_ori[3]*y_ori[3]/float(nrm)))

print(y_pos, y_ori_x, y_ori_y, y_ori_z, y_ori_w)

g = c.build_computational_graph([y_pre])
with open('graph.dot', 'w') as o:
    o.write(g.dump())

#plt.plot(x, label)
#plt.plot(x, y_pre)
#plt.show
