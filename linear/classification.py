"""
This module defines the linear classification model.
"""
import torch
import numpy as np
import utils
from torch import nn, optim


def softmax_classify(X, W, b):
    """
    X is of size (sample_num, feature_num), W is of size (feature_num, label_num), and b is of size (label_num).
    """
    return utils.softmax(torch.mm(X.view(-1, num_inputs), W) + b)


def train_softmax_classify(train_iter, test_iter, W, b, epoch_num, lr, batch_size):
    for epoch in range(epoch_num):
        train_ls_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = softmax_classify(X, W, b)
            ls = utils.cross_entropy_loss(y_hat, y).sum()
            ls.backward()

            # use sgd to update the grad
            utils.sgd([W, b], lr, batch_size)

            # if grad has not been calculated, it is NoneType
            W.grad.data.zero_()
            b.grad.data.zero_()

            train_ls_sum += ls.item()
            train_acc_sum += utils.get_classify_accuracy(y_hat, y)
            n += y.shape[0]

        # evaluate accuracy on the test set
        test_acc = utils.evaluate_classify_accuracy(test_iter, softmax_classify, W, b)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_softmax_classify(test_iter):
    X, y = iter(test_iter).next()
    true_labels = utils.get_fashion_MNIST_labels(y.numpy())
    pred_labels = utils.get_fashion_MNIST_labels(softmax_classify(X, W, b).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    utils.show_fashion_MNIST(X[0:10], titles[0:10], save_path='../figs/softmax_classify_fashion_MNIST.png')


class SoftmaxNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, X):
        # pay attention to this!
        # input is a four-tuple (sample_num, channel, width, height),
        # but in softmax regression what we need is (sample_num, feature_num)
        # this operate is called flatten
        return self.linear(X.view(X.shape[0], -1))

    def get_learnable_params(self):
        return self.parameters()


def train_softmax_net(train_iter, test_iter, net, loss, optimizer, epoch_num):
    for epoch in range(epoch_num):
        train_ls_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = net(X)
            ls = loss(y_hat, y).sum()
            optimizer.zero_grad()

            # call backward() to calculate the grad of params
            ls.backward()
            # use optimizer to update the grad for one step
            optimizer.step()

            train_ls_sum += ls.item()
            train_acc_sum += utils.get_classify_accuracy(y_hat, y)
            n += y.shape[0]

        # evaluate accuracy on the test set
        test_acc = utils.evaluate_classify_accuracy_net(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_softmax_net(test_iter, net):
    X, y = iter(test_iter).next()
    true_labels = utils.get_fashion_MNIST_labels(y.numpy())
    pred_labels = utils.get_fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    utils.show_fashion_MNIST(X[0:10], titles[0:10], save_path='../figs/softmax_net_fashion_MNIST.png')


if __name__ == '__main__':
    epoch_num = 5
    lr = 0.1
    batch_size = 256
    train_iter, test_iter = utils.load_fashion_MNIST(batch_size, resize=None, root='../data', num_workers=4)
    num_inputs, num_outputs = 784, 10
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

    # 1. test softmax classification
    train_softmax_classify(train_iter, test_iter, W, b, epoch_num, lr, batch_size)
    test_softmax_classify(test_iter)

    # 2. test SoftmaxNet
    net = SoftmaxNet(num_inputs, num_outputs)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr)

    train_softmax_net(train_iter, test_iter, net, loss, optimizer, epoch_num)
    test_softmax_net(test_iter, net)
