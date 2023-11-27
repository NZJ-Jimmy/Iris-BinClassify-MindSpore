# 2-1. 导入MindSpore模块和辅助模块

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import LossMonitor
from mindspore.ops import operations as P

# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# Windows version, set to use CPU for graph calculation

# 2-2. 读取Iris数据集，并查看部分数据
with open("iris.csv", 'r') as f:
    DictReader = csv.DictReader(f)
    iris = {'data': [], 'target': []}
    for i in DictReader:
        tdict = dict()
        if i['class'] == "Iris-setosa":
            i.pop('class')
            iris['data'].append(list(map(float, i.values())))
            iris['target'].append(0)
        elif i['class'] == "Iris-versicolor":
            i.pop('class')
            iris['data'].append(list(map(float, i.values())))
            iris['target'].append(1)

# 2-3. 抽取样本
X = iris['data'][:100]
Y = iris['target'][:100]

# 2-4. 样本可视化
setosa_len = []
versicolor_len = []
setosa_wid = []
versicolor_wid = []
for i in range(100):
    if Y[i] == 0:  # setosa
        setosa_len.append(X[i][0])
        setosa_wid.append(X[i][1])
    else:  # versicolor
        versicolor_len.append(X[i][0])
        versicolor_wid.append(X[i][1])
plt.scatter(setosa_len, setosa_wid, label='Iris-setosa')
plt.scatter(versicolor_len, versicolor_wid, label='Iris-versicolor')
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc='upper right')
plt.show()

# 2-5. 分割数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=9)

# 2-6. 数据类型转换
X_train = np.expand_dims(np.array(X_train), axis=1).astype(np.float32)  # 升为二维并转换为float32
Y_train = np.expand_dims(np.expand_dims(Y_train, axis=1),axis=1).astype(np.float32)  # 升为二维并转换为float32，以用于后面的运算
train_set = dataset.GeneratorDataset(zip(X_train, Y_train), column_names=['data', 'label'])

# 3-1. 可视化逻辑回归函数（Sigmoid）
Sigmoid_x = np.linspace(-10, 10, 100)
Sigmoid_list = 1 / (1 + np.e ** -Sigmoid_x)
plt.plot(Sigmoid_x, Sigmoid_list)
plt.show()


# 3-2 建模
class MyNet(nn.Cell):  # 定义神经网络
    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Dense(4, 1) # 线性部分

        self.fc2 = nn.Sigmoid() # 非线性部分

    def construct(self, x):
        p=self.fc1(x)
        z=self.fc2(p)
        return z

# 定义loss函数
loss_fn = P.SigmoidCrossEntropyWithLogits()

# 3-3 模型训练
epochs = 10
net = MyNet()  # 神经网络为自定义的网络
net_with_loss = nn.WithLossCell(net, loss_fn) # 定义带loss_fn的Cell
net_opt = nn.Adam(net.trainable_params(), learning_rate=0.1) # 定义优化器

model = ms.Model(net_with_loss, optimizer=net_opt) # 初始化模型
model.train(epochs, train_set, callbacks=LossMonitor()) # 开始训练

# 3-4 评估
# 构造测试集
X_test = np.expand_dims(X_test, axis=1).astype(np.float32)
test_set = dataset.GeneratorDataset(zip(X_test, Y_test), column_names=['data', 'label'])

# 评估
currect=0
total=0
for data in test_set.create_dict_iterator():
    output = net(data['data'])[0,0]
    total+=1
    if output>=0.5 and data['label']==1 or output<0.5 and data['label']==0:
        # 当模型预测值大于setosa，则为，否则为versicolor
        currect+=1

print("Accuracy: ", currect/total)
