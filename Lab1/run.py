import numpy as np
from utils import data_preprocessing
from model import RegressionModel
from utils import dataSet
import matplotlib.pyplot as plt
import argparse

# 输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="abalone.data", help='dataset_name')
parser.add_argument('--target', type=str, default="Rings", help='target')
parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='lr')
parser.add_argument('--num_of_features', type=int, default=8, help='num_of_features')
parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1')
parser.add_argument('--beta_2', type=float, default=0.99, help='beta_2')
args = parser.parse_args()

# 构建并运行模型
x_train, x_test, y_train, y_test = data_preprocessing(args.dataset_name, args.target)

train_dataset = dataSet(x_train, y_train)
test_dataset = dataSet(x_test, y_test)
model = RegressionModel(args.batch_size, args.num_of_features, args.learning_rate, args.beta_1, args.beta_2)
train_loss = model.train(train_dataset)

# 绘制折线图
x = np.arange(len(train_loss))
plt.plot(x, train_loss)
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()

# 输出最终结果
test_loss = model.predict(test_dataset)
print("test_loss:", np.mean(test_loss))
