import numpy as np
import torch
import torch.optim as optim

from logistic_regression_softmax import LogisticRegression


class Model():
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
        X = np.array(X)
        Y = np.array(y)
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)

    def train(self, epochs: int):
        losses = []  # 创建一个列表来保存每个训练轮次的损失值
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(self.X)
            loss = self.criterion(y_pred, self.y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())  # 在列表中保存当前轮次的损失值
            if (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        return losses  # 返回包含所有轮次损失值的列表

    def test(self):
        with torch.no_grad():
            output = self.model(self.X)
            y_scores = torch.nn.functional.softmax(output, dim=1)[:, 1]  # 获取预测正类的概率
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == self.y).sum().item() / self.y.size(0)
            print('Accuracy of the network on the test data: {} %'.format(100 * accuracy))
            return predicted.numpy(), self.y.numpy(), y_scores.numpy()  # 返回预测的标签、真实的标签和预测的概率分数
            
        
