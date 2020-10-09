"""
This module defines the multi-layer perceptron.
"""
import torch
import numpy as np
import tools
import metrics
from torch import nn, optim


def mlp(X):
    """
    A MLP with one hidden layer.
    X could be a 4-h_exp tensor.
    Parameters are defined outside.
    """
    X = X.view(X.shape[0], -1)
    # torch.matmul is used for tensor multiply, W1 and W2 broadcast automatically
    H = metrics.relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


class MLPNet(nn.Module):
    """
    A MLP with one hidden layer.
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(MLPNet, self).__init__()
        self.flatten = metrics.FlattenLayer()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.act = nn.ReLU()
        self.output = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        h = self.act(self.hidden(self.flatten(X)))
        return self.output(h)


def dropout_mlp(X, is_training=True):
    """
    A MLP with two hidden layers and dropout.
    """
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:
        H1 = metrics.dropout(H1, drop_prob=dropout_prob1)
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = metrics.dropout(H2, drop_prob=dropout_prob2)
    return torch.matmul(H2, W3) + b3


class DropoutMLP(nn.Module):
    """
    A MLP with two hidden layers and dropout.
    """
    def __init__(self, num_inputs, num_hiddens, num_hiddens1, num_outputs, dropout_prob1, dropout_prob2):
        super(DropoutMLP, self).__init__()
        self.hiddens = nn.Sequential(
            metrics.FlattenLayer(),

            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout_prob1),

            nn.Linear(num_hiddens, num_hiddens1),
            nn.ReLU(),
            nn.Dropout(dropout_prob2)
        )
        self.output = nn.Linear(num_hiddens1, num_outputs)

    def forward(self, X):
        h = self.hiddens(X)
        return self.output(h)


if __name__ == '__main__':
    epoch_num = 5
    lr = 80
    batch_size = 256
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size, resize=None, root='../data', num_workers=4)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    # 1. test mlp
    print('test mlp')
    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    params = [W1, b1, W2, b2]
    loss = nn.CrossEntropyLoss()
    metrics.universal_train(mlp, train_iter, test_iter, loss, epoch_num, batch_size, params=params, lr=lr, optimizer=None)
    metrics.test_classify_mnist(test_iter, mlp, save_path='../figs/mlp_classify_fashion_MNIST.png')

    # 2. test MLPNet
    print('test MLPNet')
    net = MLPNet(num_inputs, num_hiddens, num_outputs)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    metrics.universal_train(net, train_iter, test_iter, loss, epoch_num, batch_size, None, None, optimizer)
    metrics.test_classify_mnist(test_iter, net, save_path='../figs/mlp_net_fashion_MNIST.png')

    # 3. test dropout_mlp
    print('test dropout_mlp')
    num_hiddens1 = 256
    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_hiddens1)), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_hiddens1, dtype=torch.float, requires_grad=True)
    W3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_outputs)), dtype=torch.float, requires_grad=True)
    b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    params = [W1, b1, W2, b2, W3, b3]
    dropout_prob1, dropout_prob2 = 0.2, 0.5
    metrics.universal_train(dropout_mlp, train_iter, test_iter, loss, epoch_num, batch_size, params=params, lr=lr,
                            optimizer=None)

    # 4. test DropoutMLP
    print('test DropoutMLP')
    net = DropoutMLP(num_inputs, num_hiddens, num_hiddens1, num_outputs, dropout_prob1, dropout_prob2)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    metrics.universal_train(net, train_iter, test_iter, loss, epoch_num, batch_size, None, None,
                            optimizer=optimizer)

