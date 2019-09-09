import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

def squared(x):
    return pow(x,2)

n_data = 10000
dataset = np.zeros((n_data,2))

for i in range(n_data):
    x = np.random.uniform(1,10)
    dataset[i]=[x, squared(x)]

min_max = np.zeros((2,2))
min_max[0] = np.min(dataset[:], axis=0)
min_max[1] = np.max(dataset[:], axis=0)

print(min_max)
dataset = (dataset-min_max[0])/(min_max[1]-min_max[0])
dataset = dataset.astype(np.float32)

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(1, 100),
            l3=L.Linear(100, 1),
        )

    def __call__(self, x, y):
        h = F.relu(self.l1(x))
        h = self.l3(h)
        loss = F.mean_squared_error(h, y)
        report({'loss': loss}, self)
        return loss

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = self.l3(h)
        return h
from chainer import iterators, optimizers, serializers, report, training
from chainer.datasets import TupleDataset
from chainer.training import extensions

nn = MLP()

#chainer.cuda.get_device(0).use()
#nn.to_gpu()
#print("using GPU \n")

print(dataset[:9500,0:1].shape)
print(dataset[:9500,0].shape)
print(dataset[:9500].shape)
train = TupleDataset(dataset[:9500,0:1],dataset[:9500,1:2])
val = TupleDataset(dataset[9500:,0:1],dataset[9500:,1:2])

max_epoch = 10
batchsize = 40

train_iter = iterators.SerialIterator(train, batchsize)
val_iter = iterators.SerialIterator(val, batchsize, False, False)

optimizer = optimizers.Adam()
optimizer.setup(nn)

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (max_epoch, 'epoch'))

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(val_iter, nn, device=0))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

trainer.run()
nn.to_cpu()

x = np.linspace(1,10,100)
y = squared(x)

input_data = np.zeros((100,1))
for i in range(100):
    input_data[i] = x[i]

input_data = (input_data-min_max[0,0])/(min_max[1,0]-min_max[0,0])
input_data = input_data.astype(np.float32)

y_pred_norm = nn.predict(input_data)
y_pred = y_pred_norm*(min_max[1,1]-min_max[0,1])+min_max[0,1]


plt.plot(x,y)
plt.plot(x,y_pred.data)
plt.show()
