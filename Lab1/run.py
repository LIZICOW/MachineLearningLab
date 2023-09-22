import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import RegressionModel
from utils import dataSet
import matplotlib.pyplot as plt


def data_preprocessing():
    # 读取数据
    data = pd.read_csv('dataset/abalone.data', encoding='utf-8')
    data = data.replace(['M', 'F', 'I'], [0, 1, 2])

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.values.reshape(-1, 1)
    y = y.astype(np.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    return x_train, x_test, y_train, y_test


# 构建并运行模型
x_train, x_test, y_train, y_test = data_preprocessing()
train_dataset =dataSet(x_train, y_train)
test_dataset = dataSet(x_test, y_test)
model = RegressionModel()
train_loss = model.train(train_dataset)
test_loss = model.predict(test_dataset)

# 绘制折线图
x = np.arange(len(train_loss))
plt.plot(x, train_loss)
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()
