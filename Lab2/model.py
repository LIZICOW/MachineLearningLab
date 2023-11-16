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
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(self.X)
            loss = self.criterion(y_pred, self.y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

    def test(self):
        with torch.no_grad():
            output = self.model(self.X)
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == self.y).sum().item() / self.y.size(0)
            print('Accuracy of the network on the test data: {} %'.format(100 * accuracy))
