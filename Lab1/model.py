
import neuralNetwork


class RegressionModel(object):

    def __init__(self,batch_size,num_faetures,learning_rate,beta1,beta2):

        self.batch_size = batch_size
        self.w0 = neuralNetwork.Layer(num_faetures, 1)
        self.w0.set_beta(beta1,beta2)
        self.b0 = neuralNetwork.Layer(1, 1)
        self.b0.set_beta(beta1, beta2)
        self.alpha = learning_rate

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
                self.w0.update(grad[0], self.alpha)
                self.b0.update(grad[1], self.alpha)
            if self.get_loss(neuralNetwork.Constant(dataset.x), neuralNetwork.Constant(dataset.y)).data < 2.5:
                loop = False
            if epoch == 2000:
                loop = False
            if epoch % 50 == 0:
                loss_all.append(loss.data)
            epoch += 1
        return loss_all

    def predict(self, dataset):
        loss_all = []
        for x, y in dataset.iterate(self.batch_size):
            loss = self.get_loss(x, y)
            loss_all.append(loss.data)
        return loss_all
