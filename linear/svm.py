"""
This module implements a SVM for multi-label classification, trained by Fashion-MNIST.
"""
import torch
import tools
import metrics
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hinge_loss(y_hat, y, max_margin=1.):
    """
    hinge_loss = (\sum_{i=1}^N \max (0, \sum_{c \neq y} (y_hat_{i,c} - y_hat_i[y] + delta) ) ) / N
    :param y_hat: of size (batch_size, num_classes)
    :param y: of size (batch_size, 1) or (batch_size)
    :param max_margin: 1 in default
    :return: the hinge loss
    """
    y = y.view(-1)
    num_samples = len(y)
    corrects = y_hat[range(num_samples), y].unsqueeze(dim=0).T
    margins = y_hat - corrects + max_margin
    return torch.mean(torch.sum(torch.max(margins, other=torch.tensor(0.)), dim=1) - 1.)


def svm_loss(y_hat, y, max_margin=1., lambd=2e-3):
    """
    The hinge loss adding on the l2-norm.
    """
    # notice that the net[1].weight is defined outside
    return hinge_loss(y_hat, y, max_margin) + lambd * metrics.l2_penalty(net[1].weight)


def SVM():
    # input tensor is of size (batch_size, 1, 28, 28)
    return nn.Sequential(
        metrics.FlattenLayer(),
        nn.Linear(28 * 28, 10)
    )


def train_2dim_svm(X, Y, net):
    """
    Use 2-dim feature as the input of SVM to visualize its effect.
    """
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    num_samples = len(Y)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    for epoch in range(5):
        perm = torch.randperm(num_samples)
        train_ls = 0
        for i in range(num_samples):
            x = X[perm[i: i + 1]]
            y = Y[perm[i: i + 1]]
            optimizer.zero_grad()
            y_hat = net(x).squeeze()
            weight = net.weight.squeeze()

            ls = torch.mean(torch.clamp(1 - y * y_hat, min=0))
            ls += 0.01 * (weight.t() @ weight) / 2.0

            ls.backward()
            optimizer.step()

            train_ls += float(ls)
        print("Epoch: {:4d}\tloss: {}".format(epoch, train_ls / num_samples))


def visualize(X, model):
    tools.use_svg_display()
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()

    plt.savefig('../figs/svm.png')
    plt.show()


if __name__ == '__main__':
    # test hinge loss
    o = torch.tensor([[-0.39, 1.49, 4.21], [-4.61, 3.28, 1.46], [1.03, -2.37, -2.27]])
    print(o, o.shape)
    l = torch.tensor([0, 1, 2])
    c = o[range(3), l].unsqueeze(dim=0).T
    print(c)
    ms = o - c + 1
    tmp = torch.max(ms, other=torch.tensor(0.))
    print(ms, tmp)
    print(torch.mean(torch.sum(tmp, dim=1) - 1.))

    # test SVM
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, resize=None, root='../data', num_workers=4)
    net = SVM()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    # train(net, train_iter, test_iter, svm_loss, 25, optimizer)
    metrics.universal_train(net, train_iter, test_iter, svm_loss, 5, 256, None, None, optimizer)

    # test 2-dim SVM classifier
    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    X = (X - X.mean()) / X.std()
    Y[np.where(Y == 0)] = -1

    net = nn.Linear(2, 1)
    train_2dim_svm(X, Y, net)
    visualize(X, net)
