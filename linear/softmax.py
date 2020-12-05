"""
This module defines the linear classification model.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
import numpy as np
import tools
import metrics
from torch import nn, optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def softmax_classify(X):
    """
    X is of size (sample_num, feature_num), W is of size (feature_num, label_num), and b is of size (label_num).
    Parameters are defined outside.
    """
    return metrics.softmax(torch.mm(X.view(-1, num_inputs), W) + b)


def train_softmax_classify(train_iter, test_iter, W, b, epoch_num, lr, batch_size):
    print('Training on: cpu')
    for epoch in range(epoch_num):
        train_ls_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = softmax_classify(X)
            ls = metrics.cross_entropy_loss(y_hat, y).sum()
            ls.backward()

            # use sgd to update the grad
            metrics.sgd([W, b], lr, batch_size)

            # if grad has not been calculated, it is NoneType
            W.grad.data.zero_()
            b.grad.data.zero_()

            train_ls_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        # evaluate accuracy on the test set
        test_acc = metrics.evaluate_classify_accuracy(test_iter, softmax_classify)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_softmax_classify(test_iter):
    print('Testing on: cpu')
    X, y = iter(test_iter).next()
    true_labels = tools.get_fashion_MNIST_labels(y.numpy())
    pred_labels = tools.get_fashion_MNIST_labels(softmax_classify(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    tools.show_fashion_MNIST(X[0:10], titles[0:10], save_path='../figs/softmax_classify_fashion_MNIST.png')


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


def train_softmax_net(train_iter, test_iter, net, loss, device, optimizer, epoch_num):
    print('Training on:', device)
    net = net.to(device)
    for epoch in range(epoch_num):
        train_ls_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            ls = loss(y_hat, y).sum()
            optimizer.zero_grad()

            # call backward() to calculate the grad of params
            ls.backward()
            # use optimizer to update the grad for one step
            optimizer.step()

            train_ls_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        # evaluate accuracy on the test set
        test_acc = metrics.evaluate_classify_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_softmax_net(test_iter, net):
    print('Testing on:', device)
    X, y = iter(test_iter).next()
    X, y = X.to(device), y.to(device)
    true_labels = tools.get_fashion_MNIST_labels(y.cpu().numpy())
    pred_labels = tools.get_fashion_MNIST_labels(net(X).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    tools.show_fashion_MNIST(X[0:10].cpu(), titles[0:10], save_path='../figs/softmax_net_fashion_MNIST.png')


if __name__ == '__main__':
    epoch_num = 5
    lr = 0.1
    batch_size = 256
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size, resize=None, root='../data', num_workers=4)
    num_inputs, num_outputs = 784, 10

    # 1. test softmax classification
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    train_softmax_classify(train_iter, test_iter, W, b, epoch_num, lr, batch_size)
    test_softmax_classify(test_iter)
    print('\n\n')

    # 2. test SoftmaxNet
    net = SoftmaxNet(num_inputs, num_outputs)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr)

    train_softmax_net(train_iter, test_iter, net, loss, device, optimizer, epoch_num)
    test_softmax_net(test_iter, net)
