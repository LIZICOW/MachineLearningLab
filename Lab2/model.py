import numpy as np
import torch
import torch.optim as optim
import math
from logistic_regression_softmax import LogisticRegression
from sklearn.model_selection import train_test_split

class logistic_regression_Model():
    def __init__(self, input_dim: int, num_classes: int, lr: float):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.build()

    def build(self):
        self.model = LogisticRegression(self.input_dim, self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def load_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        self.X_train = torch.Tensor(X_train)
        self.X_test = torch.Tensor(X_test)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)

    def train(self, epochs: int):
        losses = []  # 创建一个列表来保存每个训练轮次的损失值
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())  # 在列表中保存当前轮次的损失值
            if (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        return losses  # 返回包含所有轮次损失值的列表

    def test(self):
        with torch.no_grad():
            output = self.model(self.X_test)
            y_scores = torch.nn.functional.softmax(output, dim=1)[:, 1]  # 获取预测正类的概率
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == self.y_test).sum().item() / self.y_test.size(0)
            print('Accuracy of the network on the test data: {} %'.format(100 * accuracy))
            return predicted.numpy(), self.y_test.numpy(), y_scores.numpy()  # 返回预测的标签、真实的标签和预测的概率分数



class NaiveBayes_log(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train  # 样本特征
        self.y_train = y_train  # 样本类别
        # 训练集样本中每个类别(二分类)的占比，即P(类别)，供后续使用
        self.P_label = {1: np.mean(y_train.values), 0: 1 - np.mean(y_train.values)}

    # 在数据集data中, 特征feature的值为value的样本所占比例
    # 用于计算P(特征|类别)、P(特征)
    def getFrequency(self, data, feature, value):
        num = len(data[data[feature] == value])  # 个数
        return num / (len(data))

    def predict(self, X_test):
        self.prediction = []  # 预测类别
        # 遍历样本
        for i in range(len(X_test)):
            x = X_test.iloc[i]  # 第i个样本
            log_P_feature_label0 = 1e-10 # log(P(特征|类别0))之和
            log_P_feature_label1 = 1e-10  # log(P(特征|类别1))之和
            log_P_feature = 2e-10  # log(P(特征))之和
            # 遍历特征
            for feature in X_test.columns:
                # 分子项，log(P(特征|类别))
                data0 = self.X_train[self.y_train.values == 0]  # 取类别为0的样本
                log_P_feature_label0 += math.log(self.getFrequency(data0, feature, x[feature]) + 1e-10)  # 加上1e-10以避免log(0)的问题
                data1 = self.X_train[self.y_train.values == 1]  # 取类别为1的样本
                log_P_feature_label1 += math.log(self.getFrequency(data1, feature, x[feature]) + 1e-10)  # 加上1e-10以避免log(0)的问题
                # 分母项，log(P(特征))
                log_P_feature += math.log(self.getFrequency(self.X_train, feature, x[feature]) + 1e-10)  # 加上1e-10以避免log(0)的问题

            # 属于每个类别的概率
            log_P_0 = log_P_feature_label0 + math.log(self.P_label[0]) - log_P_feature
            log_P_1 = log_P_feature_label1 + math.log(self.P_label[1]) - log_P_feature
            # 选出大概率值对应的类别
            self.prediction.append(1 if log_P_1 >= log_P_0 else 0)
        return self.prediction

