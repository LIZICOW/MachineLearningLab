import numpy as np
from sklearn import preprocessing
import neuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_preprocessing(name, target):
    # 读取数据
    file_name = 'dataset/' + name
    data = pd.read_csv(file_name, encoding='utf-8')
    if file_name == 'dataset/abalone.data':
        data = data.replace(['M', 'F', 'I'], [0, 1, 2])
    if file_name == 'dataset/HousingData.data':
        #将NA替换为平均值
        data = data.fillna(data.mean())
    if file_name == 'dataset/housing.data':
        #将ocean_proximity列进行one-hot编码
        data = pd.get_dummies(data, columns=['ocean_proximity'])
        #填充total_bedrooms中的缺失值，使用上一行的值进行填充
        data = data.fillna(method='pad')
        #median_house_value标准化
        data['median_house_value'] = preprocessing.scale(data['median_house_value'])

    x = data.drop(columns=[target])
    y = data[target]
    y = y.values.reshape(-1, 1)
    y = y.astype(np.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    return x_train, x_test, y_train, y_test


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
