import torch.nn as nn
import neuralNetwork
import numpy as np
from utils import dataSet


class RegressionModel(object):

    def __init__(self):

        self.batch_size = 50
        self.w0 = neuralNetwork.Layer(8, 1)
        self.b0 = neuralNetwork.Layer(1, 1)
        self.alpha = 1e-2

    def run(self, x):

        y1 = neuralNetwork.Linear(x, self.w0)
        return neuralNetwork.addBias(y1, self.b0)

    def get_loss(self, x, y):

        return neuralNetwork.meanSquareLoss(self.run(x), y)

    def train(self, dataset):
        loss_all = []
        loop = True
        epoch = 0
        while loop:
            loop = True
            for x, y in dataset.iterate(self.batch_size):
                loss = self.get_loss(x, y)
                grad = neuralNetwork.gradients(loss, [self.w0, self.b0])
                self.w0.update(grad[0], -self.alpha)
                self.b0.update(grad[1], -self.alpha)
            if self.get_loss(neuralNetwork.Constant(dataset.x), neuralNetwork.Constant(dataset.y)).data < 0.005:
                loop = False
            print(epoch, "loss:", loss.data)
            if epoch % 50 == 0:
                loss_all.append(loss.data)
            epoch += 1
        return loss_all

    def predict(self, dataset):
        loss_all = []
        for x, y in dataset.iterate_once(self.batch_size):
            loss = self.get_loss(x, y)
            loss_all.append(loss.data)
        return loss_all
