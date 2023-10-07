
import neuralNetwork
import numpy as np

class RegressionModel(object):

    def __init__(self,batch_size,num_faetures,learning_rate,beta1,beta2, episode,Adam=True):

        self.batch_size = batch_size
        self.w0 = neuralNetwork.Layer(num_faetures, 1)
        self.w0.set_beta(beta1,beta2)
        self.b0 = neuralNetwork.Layer(1, 1)
        self.b0.set_beta(beta1, beta2)
        self.alpha = learning_rate
        self.Adam = Adam
        self.episode = episode

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
            for x, y in dataset.iterate(self.batch_size):
                loss = self.get_loss(x, y)
                grad = neuralNetwork.gradients(loss, [self.w0, self.b0])
                self.w0.update(grad[0], self.alpha, Adam=self.Adam)
                self.b0.update(grad[1], self.alpha, Adam=self.Adam)
            if self.get_loss(neuralNetwork.Constant(dataset.x), neuralNetwork.Constant(dataset.y)).data < 0.1:
                loop = False
            if epoch == self.episode:
                loop = False
            loss_all.append(loss.data)
            epoch += 1
        return loss_all

    def predict(self, dataset):
        loss_all = []
        for x, y in dataset.iterate(self.batch_size):
            loss = self.get_loss(x, y)
            loss_all.append(loss.data)
        return loss_all

    def get_predict(self, dataset):
        pre = np.zeros(0)
        for x, y in dataset.iterate(self.batch_size):
            pre = np.append(pre, self.run(x).data)
        print(pre.shape)
        return pre

class twoLayerRegressionModel:
    def __init__(self,batch_size,num_faetures,learning_rate,beta1,beta2):

        self.batch_size = batch_size
        self.w0 = neuralNetwork.Layer(num_faetures, 16)
        self.w0.set_beta(beta1,beta2)
        self.b0 = neuralNetwork.Layer(1, 16)
        self.b0.set_beta(beta1, beta2)

        self.w1 = neuralNetwork.Layer(16, 1)
        self.w1.set_beta(beta1, beta2)
        self.b1 = neuralNetwork.Layer(1, 1)
        self.b1.set_beta(beta1, beta2)
        self.alpha = learning_rate

    def run(self, x):

        y1 = neuralNetwork.Linear(x, self.w0)
        y1 = neuralNetwork.addBias(y1, self.b0)
        a1 = neuralNetwork.ReLu(y1)
        y2 = neuralNetwork.Linear(a1, self.w1)
        return neuralNetwork.addBias(y2, self.b1)

    def get_loss(self, x, y):

        return neuralNetwork.meanSquareLoss(self.run(x), y)

    def train(self, dataset):
        loss_all = []
        loop = True
        epoch = 0
        while loop:
            for x, y in dataset.iterate(self.batch_size):
                loss = self.get_loss(x, y)
                grad = neuralNetwork.gradients(loss, [self.w0, self.b0])
                self.w0.update(grad[0], self.alpha)
                self.b0.update(grad[1], self.alpha)
            if self.get_loss(neuralNetwork.Constant(dataset.x), neuralNetwork.Constant(dataset.y)).data < 2.5:
                loop = False
            if epoch == 4000:
                loop = False
            loss_all.append(loss.data)
            epoch += 1
        return loss_all

    def predict(self, dataset):
        loss_all = []
        for x, y in dataset.iterate(self.batch_size):
            loss = self.get_loss(x, y)
            loss_all.append(loss.data)
        return loss_all