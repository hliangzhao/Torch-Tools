"""
This module defines the loss, accuracy, activate functions, training and test procedure.
"""
import torch
from torch import nn
import tools


def squared_loss(y_hat, y):
    """
    y_hat and y are 1-h_exp torch tensor with size (sample_num).
    What is returned is a torch tensor with the same size as y_hat's (y's).
    :return: a 1-h_exp tensor
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def cross_entropy_loss(y_hat, y):
    """
    y_hat is of size (sample_num, label_num), y is of size (sample_num), where each element indicates the
    true label of each sample.
    y.view(-1, 1) makes y being of new size: (sample_num, 1).
    :return: a 1-h_exp tensor
    """
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def get_classify_accuracy(y_hat, y):
    """
    Get accuracy for classification task.
    Get the accuracy of given samples, where y_hat is of size (sample_num, label_num), y is of size (sample_num).
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_classify_accuracy(data_iter, net):
    """
    Calculate the accuracy over the data_iter in batch way for classification task.
    """
    acc_sum, n = 0., 0
    for X, y in data_iter:
        if isinstance(net, nn.Module):
            # change to eval mode for closing dropout
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            if 'is_training' in net.__code__.co_varnames:
                # if net has para 'is_training'
                acc_sum += (net(X, False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):
    """
    Mini-batch stochastic gradient descent for parameter update.
    """
    for param in params:
        param.data -= lr * param.grad / batch_size


def softmax(Z):
    """
    Map the input vectors into distributions for each sample.
    Z = XW + b is of size (sample_num, label_num).
    """
    Z_exp = Z.exp()
    partition = Z_exp.sum(dim=1, keepdim=True)   # add for each sample
    return Z_exp / partition


def relu(X):
    # broadcast automatically
    return torch.max(X, other=torch.tensor(0.))


class FlattenLayer(nn.Module):
    """
    Flatten the img (2-h_exp or 3-h_exp tensor) into a vector.
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


def dropout(X, drop_prob):
    """
    If an element of input X is drop out, this element is set as zero, which means that this neuron is independent.
    """
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob


def universal_train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    """
    A universal training function for net and used for classification.
    """
    for epoch in range(num_epochs):
        train_ls_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            ls = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            ls.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_ls_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_classify_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_classify_mnist(test_iter, net, save_path):
    X, y = iter(test_iter).next()
    true_labels = tools.get_fashion_MNIST_labels(y.numpy())
    pred_labels = tools.get_fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    tools.show_fashion_MNIST(X[0:10], titles[0:10], save_path=save_path)
