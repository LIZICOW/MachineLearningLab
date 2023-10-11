from utils import data_preprocessing
from model import RegressionModel
from utils import dataSet
import matplotlib.pyplot as plt
import argparse
import numpy as np

# 输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="abalone.data", help='dataset_name')
#数据集名称
parser.add_argument('--target', type=str, default="Rings", help='target')
#目标变量
parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-2, help='lr')
#学习率
parser.add_argument('--num_of_features', type=int, default=8, help='num_of_features')
#自变量特征数量
parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1')
parser.add_argument('--beta_2', type=float, default=0.99, help='beta_2')
parser.add_argument('--adam', type=int, default=1, help='Adam')
parser.add_argument('--episode', type=int, default=50, help='episode')
args = parser.parse_args()

# 构建并运行模型
x_train, x_test, y_train, y_test = data_preprocessing(args.dataset_name, args.target)

train_dataset = dataSet(x_train, y_train)
test_dataset = dataSet(x_test, y_test)
model = RegressionModel(args.batch_size, args.num_of_features, args.lr, args.beta_1, args.beta_2, args.episode,Adam=args.adam)

train_loss = model.train(train_dataset)

if args.adam:
    # 绘制折线图
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.xlabel(f'Using Adam, lr={args.lr}')
    plt.ylabel('Loss')
    plt.title(f'Test loss = {np.mean(model.predict(test_dataset))}')
    plt.show()
    # plt.savefig(f'./pic/UsingAdamlr={args.lr}.png')
else:
    # 绘制折线图
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.xlabel(f'Without using Adam, lr={args.lr}')
    plt.ylabel('Loss')
    plt.title(f'Test loss = {np.mean(model.predict(test_dataset))}')
    plt.show()

# 输出最终结果
test_loss = model.predict(test_dataset)
print("test_loss:", np.mean(test_loss))

# y_pre = model.get_predict(test_dataset)
# plt.figure(figsize=(10, 10))
# # plt.plot(y_test)
# # plt.plot(model.get_predict(test_dataset))
# plt.scatter(y_test[:len(y_pre)], y_pre)
# plt.plot([-100,100],[-100,100], color='red')
# plt.ylabel('Predicted')
# plt.xlabel('Measured')
# if args.adam:
#     plt.title('Using Adam')
# else:
#     plt.title('Without Using Adam')
# plt.xlim((-3, 3))
# plt.ylim((-3, 3))
# plt.show()