"""
This module defines the linear regression model.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
import numpy as np
import tools
import metrics
from torch import nn, optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def linear_reg(X, W, b):
    """
    X and W are 2-dim torch tensor, b is a 1-dim torch tensor. Broadcast is triggered automatically.
    X is with size (sample_num, in_feature), W is with size (in_feature, 1), b is with size (1).
    """
    return torch.mm(X, W) + b


def train_linear_reg(features, labels, W, b, epoch_num, lr, batch_size):
    print('Training on: cpu')
    for epoch in range(epoch_num):
        for X, y in tools.get_data_batch(batch_size, features, labels):
            # (X, y) is a mini batch of data, where sample_num = batch_size
            y_hat = linear_reg(X, W, b)
            # according to the definition of loss, call backward() to calculate the grad of params
            ls = metrics.squared_loss(y_hat, y).sum()
            ls.backward()

            # use sgd to update the grad
            metrics.sgd([W, b], lr, batch_size)

            # clear the grad
            W.grad.data.zero_()
            b.grad.data.zero_()

        train_ls = metrics.squared_loss(linear_reg(features, W, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_ls.mean().item()))


class LinearNet(nn.Module):
    """
    The linear net, defined with torch. It is the same as linear_reg().
    """
    def __init__(self, in_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(in_features=in_feature, out_features=1)

    def forward(self, X):
        return self.linear(X)


def train_linear_net(features, labels, net, loss, optimizer, epoch_num, batch_size):
    print('Training on:', device)
    net = net.to(device)
    for epoch in range(epoch_num):
        ls = None
        for X, y in tools.get_data_batch_torch(batch_size, features, labels):
            X, y = X.to(device), y.to(device)
            ls = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            # call backward() to calculate the grad of params
            ls.backward()
            # use optimizer to update the grad for one step
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch + 1, ls.item()))


if __name__ == '__main__':
    # generate training set
    sample_num = 1000
    feature_num = 2
    true_W, true_b = [2, -3.4], 4.2
    features = torch.randn(sample_num, feature_num, dtype=torch.float32)
    labels = true_W[0] * features[:, 0] + true_W[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

    # 1. test linear regression
    # define params
    W = torch.tensor(np.random.normal(0, 0.01, (feature_num, 1)), dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    # train
    train_linear_reg(features, labels, W, b, epoch_num=10, lr=0.01, batch_size=20)
    print(true_W, W)
    print(true_b, b)
    print('\n\n')

    # 2. test LinearNet
    net = LinearNet(in_feature=2)
    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train_linear_net(features, labels, net, loss, optimizer, epoch_num=10, batch_size=20)
    print(true_W, net.linear.weight)
    print(true_b, net.linear.bias)
