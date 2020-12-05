"""
This module tests under-fitting, over-fitting, and regularization with the example of regression.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import tools
import metrics
from linear import regression
from torch import nn, optim
import torch
import random
import numpy as np


def generate_poly(n_train, n_test, h_exp):
    """
    Generate a polynomial with given dim and random w, b.
    h_exp is the highest exp number, e.g, h_exp = 3 --> f(x) = w1*x + w2*x^2 + w3*x^3 + b + eps.
    """
    assert h_exp > 0
    true_w = -10 + 20 * torch.rand(h_exp)
    true_b = -10 + 20 * random.random()
    features = torch.randn((n_train + n_test, 1))
    poly_features = features
    for i in range(h_exp - 1):
        tmp = torch.pow(features, i + 2)
        poly_features = torch.cat((poly_features, tmp), dim=1)
    labels = true_w[0] * poly_features[:, 0] + true_b
    for i in range(h_exp - 1):
        labels += true_w[i + 1] * poly_features[:, i + 1]
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    return true_w, true_b, features, poly_features, labels


def test_fitting(train_features, test_features, train_labels, test_labels, epoch_num, batch_size, save_path):
    """
    Use regression to evaluate the effect of model complexity and size of training set.
    The input features should be of size (sample_num, feature_num).
    """
    net = regression.LinearNet(train_features.shape[-1])
    optimizer = optim.SGD(net.parameters(), lr=0.015)
    train_ls, test_ls = [], []
    for _ in range(epoch_num):
        for X, y in tools.get_data_batch(batch_size, train_features, train_labels):
            # turn y into a 2-dim tensor! this operate is important
            y_hat = net(X)
            ls = loss(y_hat, y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        # turn y (true labels) into a 2-dim tensor! this operate is important
        train_ls.append(loss(net(train_features), train_labels.view(-1, 1)).item())
        test_ls.append(loss(net(test_features), test_labels.view(-1, 1)).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    tools.plot_semilogy(save_path,
                        range(1, epoch_num + 1), train_ls, 'epoch', 'loss',
                        range(1, epoch_num + 1), test_ls, ['train', 'test'])
    print('trained weight:', net.linear.weight.data)
    print('trained bias:', net.linear.bias.data)


def test_weight_decay(train_features, test_features, train_labels, test_labels, epoch_num, batch_size, lambd, lr, save_path):
    W = torch.randn((train_features.shape[-1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    train_ls, test_ls = [], []
    for _ in range(epoch_num):
        for X, y in tools.get_data_batch(batch_size, train_features, train_labels):
            y_hat = regression.linear_reg(X, W, b)
            ls = (metrics.squared_loss(y_hat, y) + lambd * metrics.l2_penalty(W)).sum()
            if W.grad is not None:
                W.grad.data.zero_()
                b.grad.data.zero_()
            ls.backward()
            metrics.sgd([W, b], lr, batch_size)
        train_ls.append(metrics.squared_loss(regression.linear_reg(train_features, W, b), train_labels).mean().item())
        test_ls.append(metrics.squared_loss(regression.linear_reg(test_features, W, b), test_labels).mean().item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    tools.plot_semilogy(save_path,
                        range(1, epoch_num + 1), train_ls, 'epoch', 'loss',
                        range(1, epoch_num + 1), test_ls, ['train', 'test'])
    print('trained weight:', W, 'L2-norm of it:', W.norm().item())
    print('trained bias:', b)


def test_weight_decay_torch(train_features, test_features, train_labels, test_labels, epoch_num, batch_size, lambd, save_path):
    net = regression.LinearNet(train_features.shape[-1])
    optimizer_W = optim.SGD(params=[net.linear.weight], lr=0.003, weight_decay=lambd)
    optimizer_b = optim.SGD(params=[net.linear.bias], lr=0.003)
    train_ls, test_ls = [], []
    for _ in range(epoch_num):
        for X, y in tools.get_data_batch(batch_size, train_features, train_labels):
            ls = loss(net(X), y).mean()
            optimizer_W.zero_grad()
            optimizer_b.zero_grad()
            ls.backward()
            optimizer_W.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    tools.plot_semilogy(save_path,
                        range(1, epoch_num + 1), train_ls, 'epoch', 'loss',
                        range(1, epoch_num + 1), test_ls, ['train', 'test'])
    print('trained weight:', net.linear.weight.data, 'L2-norm of it:', net.linear.weight.norm().item())
    print('trained bias:', net.linear.bias.data)


if __name__ == '__main__':
    n_train, n_test = 100, 100
    h_exp = 3
    true_w, true_b, features, poly_features, labels = generate_poly(n_train, n_test, h_exp)
    print(features.shape, poly_features.shape, labels.shape)
    print('true weight:', true_w)
    print('true bias:', true_b)

    # 1. test effects of model complexity
    loss = nn.MSELoss()
    test_fitting(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/over_fitting.png')
    test_fitting(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/under_fitting.png')

    # 2. test effects of training dataset size
    test_fitting(poly_features[:2, :], poly_features[n_train:, :], labels[:2], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/train_size.png')

    # construct over-fitting env
    n_train, n_test, num_inputs = 20, 100, 200
    # y = 0.05 + \sum_{i=1}^{200} 0.01 x_i + eps
    true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
    features = torch.randn((n_train + n_test, num_inputs))
    labels = torch.matmul(features, true_w) + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]
    print(test_features.shape, test_labels.shape)
    print('true weight:', true_w)
    print('true bias:', true_b)

    # 3. test regularization
    test_weight_decay(train_features, test_features, train_labels, test_labels,
                      epoch_num=100, batch_size=1, lambd=0, lr=0.003, save_path='../figs/little_weight_decay.png')
    test_weight_decay(train_features, test_features, train_labels, test_labels,
                      epoch_num=100, batch_size=1, lambd=4, lr=0.003, save_path='../figs/middle_weight_decay.png')
    test_weight_decay(train_features, test_features, train_labels, test_labels,
                      epoch_num=100, batch_size=1, lambd=10, lr=0.003, save_path='../figs/large_weight_decay.png')

    # 4. test regularization with torch
    # loss = nn.MSELoss()
    test_weight_decay_torch(train_features, test_features, train_labels, test_labels,
                            epoch_num=100, batch_size=1, lambd=0, save_path='../figs/little_weight_decay_torch.png')
    test_weight_decay_torch(train_features, test_features, train_labels, test_labels,
                            epoch_num=100, batch_size=1, lambd=4, save_path='../figs/middle_weight_decay_torch.png')
    test_weight_decay_torch(train_features, test_features, train_labels, test_labels,
                            epoch_num=100, batch_size=1, lambd=10, save_path='../figs/large_weight_decay_torch.png')
