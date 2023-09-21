# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd

def data_preprocessing():
    '''数据预处理'''
    # 读取数据
    data=pd.read_csv('dataset/abalone.data',encoding='utf-8')
    # 查看数据信息
    print(data.head())
    print("数据维度：",data.shape)

    # 将第一列是性别的数据转换为数值型数据
    data=data.replace(['M','F','I'],[0,1,2])
    print(data.head())

    # 将数据划分为输入数据和输出数据
    # 最后一列是输出数据 Ring
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    print(x.head())
    print(y.head())

    # 将数据划分为训练集和测试集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    print("训练集维度：",x_train.shape)
    print("测试集维度：",x_test.shape)

    # 标准化
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    x_train=std.fit_transform(x_train)
    x_test=std.transform(x_test)
    print(x_train[:5])
    print(x_test[:5])

    return x_train,x_test,y_train,y_test



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_preprocessing()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
