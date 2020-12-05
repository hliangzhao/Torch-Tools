"""
This module defines the loss, accuracy, activate functions, training and test procedure.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import tools
import time


def squared_loss(y_hat, y):
    """
    y_hat and y can be 1-dim torch tensor with size (sample_num) or 2-dim tensor with size (sample_num, 1).
    What is returned is a torch tensor with the same size as y_hat's (y's).
    :return: has the same size as y_hat
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def cross_entropy_loss(y_hat, y):
    """
    y_hat is of size (sample_num, label_num), where for each sample, the 1-dim tensor is the prob. distribution of labels.
    y is of size (sample_num) or (sample_num, 1), where each element indicates the true label of each sample.
    y.view(-1, 1) makes y being of new size: (sample_num, 1).
    """
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def hinge_loss(y_hat, y, device, max_margin=1.):
    """
    hinge_loss = (\sum_{i=1}^N \max (0, \sum_{c \neq y} (y_hat_{i,c} - y_hat_i[y] + delta) ) ) / N.
    :param y_hat: of size (batch_size, num_classes)
    :param y: of size (batch_size, 1) or (batch_size)
    :param device:
    :param max_margin: 1 in default
    :return: the hinge loss
    """
    y = y.view(-1)
    num_samples = len(y)
    corrects = y_hat[range(num_samples), y].unsqueeze(dim=0).T
    margins = y_hat - corrects + max_margin
    return torch.mean(torch.sum(torch.max(margins, other=torch.tensor(0.).to(device)), dim=1) - 1.)


def get_classify_accuracy(y_hat, y):
    """
    Get the accuracy of given samples, where y_hat is of size (sample_num, label_num), y is of size (sample_num, 1).
    :return: a scalar of accuracy for given samples
    """
    # .item() can be used only when the tensor is of size (1)
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_classify_accuracy(data_iter, net, device=None):
    """
    Calculate the accuracy over the data_iter (in batch way) for classification task.
    :return: a scalar of accuracy for given samples
    """
    if device is None and isinstance(net, nn.Module):
        # use the device of net's device
        device = list(net.parameters())[0].device
    acc_sum, n = 0., 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, nn.Module):
                # change to eval mode
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
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
    partition = Z_exp.sum(dim=1, keepdim=True)   # keep dim, each for a sample
    return Z_exp / partition


def relu(X):
    # broadcast tensor([0.]) automatically
    return torch.max(X, other=torch.tensor(0.))


class FlattenLayer(nn.Module):
    """
    Flatten the input tensor into a vector.
    The first dim of input is sample_num.
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


def dropout(X, drop_prob):
    """
    If an element of input X is dropped out, this element is set as zero, which means that this neuron is inactive.
    Because the neurons are set as zero randomly, the net could not overly rely on any neuron.
    """
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob


def l2_penalty(W):
    """
    The L2-norm of given parameter W.
    :return: a scalar of l2-norm
    """
    return (W**2).sum() / 2


def universal_train(net, train_iter, test_iter, loss, num_epochs, batch_size, device, params=None, lr=None, optimizer=None):
    """
    A universal training function for net and used for classification.
    """
    if isinstance(net, nn.Module):
        net = net.to(device)
        print('Training on:', device)
    else:
        print('Training on: cpu')
    for epoch in range(num_epochs):
        train_ls_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            if isinstance(net, torch.nn.Module):
                X, y = X.to(device), y.to(device)
            y_hat = net(X)
            ls = loss(y_hat, y).sum()

            if optimizer is not None:
                # use torch
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            ls.backward()
            if optimizer is None:
                # not use torch
                sgd(params, lr, batch_size)
            else:
                # use torch
                optimizer.step()

            train_ls_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_classify_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


def test_classify_mnist(test_iter, net, device, save_path):
    """
    Use Fashion-MNIST dataset to test the classifier's effect.
    """
    if isinstance(net, nn.Module):
        net = net.to(device)
        print('Testing on:', device)
    X, y = iter(test_iter).next()
    if isinstance(net, nn.Module):
        X, y = X.to(device), y.to(device)
    true_labels = tools.get_fashion_MNIST_labels(y.cpu().numpy())
    pred_labels = tools.get_fashion_MNIST_labels(net(X).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    tools.show_fashion_MNIST(X[0:10].cpu(), titles[0:10], save_path=save_path)


def corr1d(X, K):
    """
    The 1-dim correlation computation for X and K.
    :param X: the input feature
    :param K: the filter
    :return: the output feature
    """
    w = K.shape[0]
    Y = torch.zeros(X.shape[0] - w + 1)
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr2d(X, K):
    """
    The 2-dim correlation computation for X and K.
    :param X: the input feature
    :param K: the filter
    :return: the output feature
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # element-wise multiplication and sum afterward
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    return torch.stack([corr1d(x, k) for x, k in zip(X, K)]).sum(dim=0)


def corr2d_multi_in(X, K):
    """
    The 2-dim correlation computation for X and K with multiple in channels.
    :param X: the 3-dim input feature (in_channel, height, width)
    :param K: a 3-dim filter (in_channel, kernel_height, kernel_width)
    :return: the output feature (out_height, out_width)
    """
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


def corr2d_multi_in_multi_out(X, K):
    """
    The 2-dim correlation computation for X and K with multiple in channels and multiple out channels.
    :param X: the 3-dim input feature (in_channel, height, width)
    :param K: the 4-dim filter (in_channel, out_channel, kernel_height, kernel_width)
    :return: the output feature (out_channel, out_height, out_width)
    """
    return torch.stack([corr2d_multi_in(X, k) for k in K])


def corr2d_1x1(X, K):
    """
    The 1x1 conv layer.
    This layer is similar to fully connected layer.
    :param X: the 3-dim input feature (in_channel, height, width)
    :param K: the 4-dim filter (in_channel, out_channel, 1, 1)
    :return: the output feature (out_channel, out_height, out_width)
    """
    assert tuple(K.shape[2:]) == (1, 1)
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


def train_kernel2D(X, Y, epoch_num, lr, conv_layer):
    """
    Learn filter from data.
    """
    for i in range(epoch_num):
        Y_hat = conv_layer(X)
        ls = ((Y_hat - Y) ** 2).sum()
        ls.backward()

        conv_layer.weight.data -= lr * conv_layer.weight.grad
        conv_layer.bias.data -= lr * conv_layer.bias.grad

        conv_layer.weight.grad.fill_(0)
        conv_layer.bias.grad.fill_(0)

        if (i + 1) % 5 == 0:
            print('Epoch %d, loss %f' % (i + 1, ls.item()))


def comp_conv2d(conv_layer, X):
    """
    Get the output feature. X is a 2-dim torch tensor (both are features).
    """
    # in_channel, out_channel, height, width
    X = X.view((1, 1) + X.shape)
    Y = conv_layer(X)
    return Y.view(Y.shape[2:])


def pool2d(X, pool_size, mode='max'):
    """
    The (max or avg) pooling layer.
    :param X: the 2-dim input feature (height, width)
    :param pool_size: the tuple (p_h, p_w)
    :param mode: 'max' or 'avg'
    :return: the output feature (out_height, out_width)
    """
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


class GlobalAvgPool2d(nn.Module):
    """
    Calculate the average value of each channel as one output.
    The output units num is the number of in_channels.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, X):
        return F.avg_pool2d(X, kernel_size=X.size()[2:])


class GlobalMaxPool1d(nn.Module):
    """
    Max-over-time pooling.
    The return of each channel is the maximum element (over num_steps) of this channel.
    """
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, X):
        # X is of size (batch_size, channel_num, num_steps)
        # return (batch_size, channel_num, 1)
        return F.max_pool1d(X, kernel_size=X.shape[2])


def cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs):
    """
    A universal training function for CNN and used for classification.
    """
    net = net.to(device)
    print('training on', device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_ls_sum, train_acc_sum, n, batch_count, start = 0., 0., 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            ls = loss(y_hat, y)
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            train_ls_sum += ls.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            # print('batch %d finished' % batch_count)
        test_acc = evaluate_classify_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_ls_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def onehot_encoding(X, num_classes, dtype=torch.float32):
    """
    Get one-hot encoding of given input (features).
    :param X: of size (batch_size)
    :param num_classes:
    :param dtype:
    :return: of size (batch_size, num_classes)
    """
    X = X.long()
    res = torch.zeros((X.shape[0], num_classes), dtype=dtype, device=X.device)
    # for each row of res, fill 1 into the position remarked by X
    # (take X as the indices of position for each sample in the batch)
    res.scatter_(dim=1, index=X.view(-1, 1), value=1)
    return res


def to_onehot(X, num_classes):
    """
    Get one-hot encoding of given data batch X.
    :param X: of size (batch_size, num_steps)
    :param num_classes:
    :return: several tensor of size (batch_size, num_classes), here 'several' is actually num_steps
    """
    return [onehot_encoding(X[:, i], num_classes) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    """
    Clip gradient when they are too large.
    g' <--- min (theta / l2-norm(g), 1) x g.
    :param params:
    :param theta:
    :param device:
    :return:
    """
    norm = torch.tensor([0.], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= theta / norm


if __name__ == '__main__':
    # test Conv2D
    X = torch.ones(6, 8)
    X[:, 2: 6] = 0
    K = torch.tensor([[-1, 1]])
    Y = corr2d(X, K)
    conv2d = Conv2D((1, 2))
    train_kernel2D(X, Y, epoch_num=30, lr=0.01, conv_layer=conv2d)
    print(conv2d.weight.data, conv2d.bias.data)

    # test get_conv2d_out_size
    X = torch.randn(8, 8)
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1), stride=1)
    print(comp_conv2d(conv2d, X).shape)

    # test corr2d_multi_in
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]],
                      [[1, 2], [3, 4]]])
    print(X.shape, K.shape, corr2d_multi_in(X, K).shape)

    # test corr2d_multi_in_multi_out
    K = torch.stack([K, K + 1, K + 2])
    print(X.shape, K.shape, corr2d_multi_in_multi_out(X, K).shape)

    # test nn.MaxPool2d
    X = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
    pool2d_layer = nn.MaxPool2d(3)
    print(pool2d_layer(X))
    pool2d_layer = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d_layer(X))
    pool2d_layer = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
    print(pool2d_layer(X))

    # test multi-in_channel
    X = torch.cat((X, X + 1), dim=1)
    pool2d_layer = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
    print(pool2d_layer(X))

    # test one-hot encoding
    X = torch.tensor([0, 2])
    print(onehot_encoding(X, num_classes=4))
    X = torch.arange(10).view(2, 5)
    inputs = to_onehot(X, num_classes=10)
    print(X, '\n', len(inputs), '\n', inputs[0])
