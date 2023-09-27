import numpy as np
import neuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def data_preprocessing(name, target):
    # 读取数据
    file_name = 'dataset/' + name
    data = pd.read_csv(file_name, encoding='utf-8')
    data = data.replace(['M', 'F', 'I'], [0, 1, 2])

    x = data.drop(columns=[target])
    y = data[target]
    y = y.values.reshape(-1, 1)
    y = y.astype(np.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    return x_train, x_test, y_train, y_test


def show_graph(train_loss, y_predict, y_test):
    # 绘制折线图
    plt.plot(range(len(train_loss)), train_loss)
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()


    # 绘制全部图像对比
    plt.figure(figsize=(8, 6))
    # 创建两个子图，并共享 x 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(range(len(y_predict)), y_predict, label='predict', color='blue')
    ax2.plot(range(len(y_test)), y_test[:-4], label='groundtruth', color='red')
    fig.suptitle('Comparison of predict and truth')
    # 显示图表
    plt.show()


class dataSet:
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate(self, batch_size):
        num_batches = self.x.shape[0] // batch_size  # 计算可完整迭代的批次数
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x = self.x[start_idx:end_idx]
            y = self.y[start_idx:end_idx]
            yield neuralNetwork.Constant(x), neuralNetwork.Constant(y)
