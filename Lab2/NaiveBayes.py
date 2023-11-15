import numpy as np
import pandas as pd

class NaiveBayes(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train  #样本特征
        self.y_train = y_train  #样本类别
        #训练集样本中每个类别(二分类)的占比，即P(类别)，供后续使用
        self.P_label = {1: np.mean(y_train.values), 0: 1-np.mean(y_train.values)}

    #在数据集data中, 特征feature的值为value的样本所占比例
    #用于计算P(特征|类别)、P(特征)
    def getFrequency(self, data, feature, value):
        num = len(data[data[feature]==value]) #个数
        return num / (len(data))

    def predict(self, X_test):
        self.prediction = [] #预测类别
        # 遍历样本
        for i in range(len(X_test)):
            x = X_test.iloc[i]      # 第i个样本
            P_feature_label0 = 1    # P(特征|类别0)之和
            P_feature_label1 = 1    # P(特征|类别1)之和
            P_feature = 1           # P(特征)之和
            # 遍历特征
            for feature in X_test.columns:
                # 分子项，P(特征|类别)
                data0 = self.X_train[self.y_train.values==0]  #取类别为0的样本
                P_feature_label0 *= self.getFrequency(data0, feature, x[feature]) #计算P(feature|0)

                data1 = self.X_train[self.y_train.values==1]  #取类别为1的样本
                P_feature_label1 *= self.getFrequency(data1, feature, x[feature]) #计算P(feature|1)

                # 分母项，P(特征)
                P_feature *= self.getFrequency(self.X_train, feature, x[feature])

            #属于每个类别的概率
            P_0 = (P_feature_label0*self.P_label[0]) / P_feature
            P_1 = (P_feature_label1 * self.P_label[1]) / P_feature
            #选出大概率值对应的类别
            self.prediction.append(1 if P_1>=P_0 else 0)
        return self.prediction