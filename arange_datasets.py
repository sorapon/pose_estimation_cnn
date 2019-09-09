import numpy as np
import argparse
import h5py
from chainer import serializers
import network_1 as network
import time
import random
from tqdm import tqdm

## 3d show
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--num_dataset', '-n', default=23500, type=int, #default=-1
                    help='the number of division for each axis')
parser.add_argument('--division', '-d', default=50, type=int,
                    help='the number of division for each axis')

args = parser.parse_args()
#xp = cuda.cupy if args.gpu >= 0 else np
n_data = args.num_dataset
div = args.division
t0 = time.time()

channel = 1
axis_x = div
axis_y = div
axis_z = div

x = np.zeros((n_data*4, channel, axis_z, axis_y, axis_x), dtype='int8')
y = np.zeros((n_data*4,7))

print("load dataset")
infh = h5py.File('./datasets/hv8_70_14000.hdf5', 'r')
infh.keys()

print(x.shape)
print(y.shape)

for n in tqdm(range(0,n_data)):
    voxel_data = infh["data_"+str(n+1)]['voxel'].value
    tf_data = infh["data_"+str(n+1)]['pose'].value
    y[n] = tf_data

#### visualize voxel data
##    point_x = []
##    point_y = []
##    point_z = []
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##    ax.set_xlabel("x")
##    ax.set_ylabel("y")
##    ax.set_zlabel("z")

    for m in range(channel):
        for i in range(axis_z):
            for j in range(axis_y):
                for k in range(axis_x):
                    if(voxel_data[(div*div*i) + (div*j) + (k)] == 1):
                        x[n+n_data*0, m, i, j, k] = 1
                        x[n+n_data*1, m, i, j, k] = 1
                        x[n+n_data*2, m, i, j, k] = 1
                        x[n+n_data*3, m, i, j, k] = 1

    # 欠損値の大きさと原点
    van_len_x = random.randint(5, 20)
    van_len_y = random.randint(5, 20)
    van_len_z = random.randint(5, 20)
    van_ori_x = random.randint(0, 30)
    van_ori_y = random.randint(0, 30)
    van_ori_z = random.randint(0, 30)

    # 欠損値の大きさと原点
    len_1_x = random.randint(5, 20)
    len_1_y = random.randint(5, 20)
    len_1_z = random.randint(5, 20)
    ori_1_x = random.randint(0, 30)
    ori_1_y = random.randint(0, 30)
    ori_1_z = random.randint(0, 30)

    # box1の大きさと原点
    len_1_x = random.randint(5, 20)
    len_1_y = random.randint(5, 20)
    len_1_z = random.randint(5, 20)
    ori_1_x = random.randint(0, 30)
    ori_1_y = random.randint(0, 30)
    ori_1_z = random.randint(0, 30)

    # box2の大きさと原点
    rand = random.randint(1, 2)
    if (rand == 2):
       len_2_x = random.randint(5, 20)
       len_2_y = random.randint(5, 20)
       len_2_z = random.randint(5, 20)
       ori_2_x = random.randint(0, 30)
       ori_2_y = random.randint(0, 30)
       ori_2_z = random.randint(0, 30)

    # 欠損値処理
    x[n+n_data*1, m, van_ori_z:van_ori_z + van_len_z, van_ori_y:van_ori_y+van_len_y, van_ori_x:van_ori_x+van_len_x] = 0
    x[n+n_data*3, m, van_ori_z:van_ori_z + van_len_z, van_ori_y:van_ori_y+van_len_y, van_ori_x:van_ori_x+van_len_x] = 0

    # box1付加
    x[n+n_data*2, m, ori_1_z:ori_1_z+len_1_z, ori_1_y:ori_1_y+len_1_y, ori_1_x:ori_1_x+len_1_x] = 1
    x[n+n_data*3, m, ori_1_z:ori_1_z+len_1_z, ori_1_y:ori_1_y+len_1_y, ori_1_x:ori_1_x+len_1_x] = 1
    # box2付加
    if (rand == 2):
       x[n+n_data*2, m, ori_2_z:ori_2_z+len_2_z, ori_2_y:ori_2_y+len_2_y, ori_2_x:ori_2_x+len_2_x] = 1
       x[n+n_data*2, m, ori_2_z+1:ori_2_z+len_2_z-1, ori_2_y+1:ori_2_y+len_2_y-1, ori_2_x+1:ori_2_x+len_2_x-1] = 0
       x[n+n_data*3, m, ori_2_z:ori_2_z+len_2_z, ori_2_y:ori_2_y+len_2_y, ori_2_x:ori_2_x+len_2_x] = 1
       x[n+n_data*3, m, ori_2_z+1:ori_2_z+len_2_z-1, ori_2_y+1:ori_2_y+len_2_y-1, ori_2_x+1:ori_2_x+len_2_x-1] = 0

    x[n+n_data*2, m, ori_1_z+1:ori_1_z+len_1_z-1, ori_1_y+1:ori_1_y+len_1_y-1, ori_1_x+1:ori_1_x+len_1_x-1] = 0
    x[n+n_data*3, m, ori_1_z+1:ori_1_z+len_1_z-1, ori_1_y+1:ori_1_y+len_1_y-1, ori_1_x+1:ori_1_x+len_1_x-1] = 0

    save_voxel = np.ravel(x)
##                        point_x.append(k)
##                        point_y.append(j)
##                        point_z.append(i)
##
##    ax.scatter(point_x, point_y, point_z)
##    plt.show()
##        print(x[n][m])

y[n_data:n_data*2] = y[0:n_data]
y[n_data*2:n_data*3] = y[0:n_data]
y[n_data*3:n_data*4] = y[0:n_data]

f_hdf5 = h5py.File('./datasets/hv8_70_48000.hdf5','w')

for i in range(0, n_data*4):
    data_group = f_hdf5.create_group("data_"+str(i+1))

    data_group.create_dataset("voxel", data=np.ravel(x[i]), compression= "lzf")
    data_group.create_dataset("pose", data=y[i], compression= "lzf")


t1 = time.time()
elapsed_time1 = t1-t0
print("データ読み込み時間：{}".format(elapsed_time1))
