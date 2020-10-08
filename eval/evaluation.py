import tools
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


def fit_and_plot(train_features, test_features, train_labels, test_labels, epoch_num, batch_size, save_path):
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


if __name__ == '__main__':
    n_train, n_test = 100, 100
    h_exp = 3
    true_w, true_b, features, poly_features, labels = generate_poly(n_train, n_test, h_exp)
    print(features.shape, poly_features.shape, labels.shape)
    print('true weight:', true_w)
    print('true bias:', true_b)

    # 1. test effects of model complexity
    loss = nn.MSELoss()
    fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/over_fitting.png')
    fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/under_fitting.png')

    # 2. test effects of training dataset size
    fit_and_plot(features[:2, :], features[n_train:, :], labels[:2], labels[n_train:],
                 epoch_num=100, batch_size=10, save_path='../figs/train_size.png')
