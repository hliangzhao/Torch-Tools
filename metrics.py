"""
This module defines the loss, accuracy, activate functions, training and test procedure.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import tools
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from linear.regression import linear_reg


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


def hinge_loss(y_hat, y, max_margin=1.):
    """
    hinge_loss = (\sum_{i=1}^N \max (0, \sum_{c \neq y} (y_hat_{i,c} - y_hat_i[y] + delta) ) ) / N.
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


def get_classify_accuracy(y_hat, y):
    """
    Get the accuracy of given samples, where y_hat is of size (sample_num, label_num), y is of size (sample_num, 1).
    :return: a scalar of accuracy for given samples
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_classify_accuracy(data_iter, net, device=None):
    """
    Calculate the accuracy over the data_iter in batch way for classification task.
    :return: a scalar of accuracy for given samples
    """
    if device is None and isinstance(net, nn.Module):
        # use the device of net's device
        device = list(net.parameters())[0].device
    acc_sum, n = 0., 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, nn.Module):
                # change to eval mode for closing dropout
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
    partition = Z_exp.sum(dim=1, keepdim=True)   # add for each sample
    return Z_exp / partition


def relu(X):
    # broadcast automatically
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
    If an element of input X is dropped out, this element is set as zero, which means that this neuron is independent.
    (Set the value of neurons as zero randomly.)
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


def test_classify_mnist(test_iter, net, save_path):
    """
    Use Fashion-MNIST dataset to test the classifier's effect.
    """
    X, y = iter(test_iter).next()
    true_labels = tools.get_fashion_MNIST_labels(y.numpy())
    pred_labels = tools.get_fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    tools.show_fashion_MNIST(X[0:10], titles[0:10], save_path=save_path)


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
            # element-wise multiplication and adding afterward
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


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


def rnn_predict(prefix, num_chars, rnn, params, init_hidden_state, num_hiddens,
                vocab_size, idx_to_char, char_to_idx, device):
    """
    For given prefix of chars, predict the next num_chars chars.
    """
    hidden_state = init_hidden_state(batch_size=1, num_hiddens=num_hiddens, device=device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # batch_size, num_steps = 1, 1
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, hidden_state) = rnn(X, hidden_state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # notice that here Y is a list of 1 tensor, which is of size (1, vocab_size)
            # thus we need Y[0] to get this tensor and compare the value on dim 1
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[o] for o in output])


def rnn_predict_torch(prefix, num_chars, model, idx_to_char, char_to_idx, device):
    """
    For given prefix of chars, predict the next num_chars chars.
    """
    hidden_state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if hidden_state is not None:
            if isinstance(hidden_state, tuple):
                hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
            else:
                hidden_state = hidden_state.to(device)
        (Y, hidden_state) = model(X, hidden_state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # notice that here Y is of size (1, vocab_size) because both num_steps and batch_size are 1
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


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


def rnn_train_and_predict(model, params, init_hidden_state, num_hiddens, vocab_size, idx_to_char, char_to_idx, device,
                          corpus_indices, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    """
    RNN train and prediction.
    :param model: the fn to get a rnn model
    :param params:
    :param init_hidden_state:
    :param num_hiddens:
    :param vocab_size:
    :param idx_to_char:
    :param char_to_idx:
    :param device:
    :param corpus_indices:
    :param is_random_iter: the data batch is obtained by random sampling or consecutive sampling
    :param num_epochs:
    :param num_steps:
    :param lr:
    :param clipping_theta:
    :param batch_size:
    :param pred_period: how often do we make predictions
    :param pred_len: the next pred_len chars to predict
    :param prefixes: the list of prefixes
    :return:
    """
    if is_random_iter:
        data_iter_fn = tools.get_timeseries_data_batch_random
    else:
        data_iter_fn = tools.get_timeseries_data_batch_consecutive
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            # if use consecutive sampling, the hidden state should be initialized only at the beginning of each epoch
            # initialization is not required at the beginning of each minibatch
            hidden_state = init_hidden_state(batch_size, num_hiddens, device)
        ls_sum, n, start = 0., 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                # if use random sampling, the hidden state should be initialized at the beginning of each minibatch
                hidden_state = init_hidden_state(batch_size, num_hiddens, device)
            else:
                # notice that hidden_state is of size (batch_size, num_hiddens), which are belated from the last epoch
                # the hidden states of these samples should be detached from the computation graph
                for s in hidden_state:
                    s.detach_()

            inputs = to_onehot(X, num_classes=vocab_size)
            (outputs, hidden_state) = model(inputs, hidden_state, params)
            # notice that the outputs is a list of num_steps tensors, each of size (batch_size, vocab_size)
            # after torch.cat, outputs is a tensor of size (num_steps * batch_size, vocab_size)
            # outputs = [sample1_step1, sample2_step1, ..., samplen_step1,
            #            sample1_step2, sample2_step2, ..., samplen_step2,
            #            ...        ...     ...     ...
            #            sample1_stepm, sample2_stepm, ..., samplen_stepm]
            outputs = torch.cat(outputs, dim=0)
            # y = [sample1_step1, sample2_step1, ..., samplen_step1,
            #      sample1_step2, sample2_step2, ..., samplen_step2,
            #      ...        ...     ...     ...
            #      sample1_stepm, sample2_stepm, ..., samplen_stepm]
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            ls = loss(outputs, y.long())    # turn y into long type as label

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            ls.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, batch_size=1)         # ls has been averaged, here batch_size is set as 1
            ls_sum += ls.item() * y.shape[0]
            n += y.shape[0]                       # y.shape[0] is num_steps * batch_size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(ls_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', rnn_predict(prefix, pred_len, model, params, init_hidden_state, num_hiddens,
                                        vocab_size, idx_to_char, char_to_idx, device))


def rnn_train_and_predict_torch(model, vocab_size, idx_to_char, char_to_idx, device,
                                corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    hidden_state = None
    for epoch in range(num_epochs):
        ls_sum, n, start = 0., 0, time.time()
        data_iter = tools.get_timeseries_data_batch_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if hidden_state is not None:
                if isinstance(hidden_state, tuple):
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                else:
                    hidden_state = hidden_state.detach()
            (output, hidden_state) = model(X, hidden_state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            ls = loss(output, y.long())                 # output and y are of shape (num_steps * batch_size, vocab_size)
            optimizer.zero_grad()
            ls.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            ls_sum += ls.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(ls_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', rnn_predict_torch(prefix, pred_len, model, idx_to_char, char_to_idx, device))


def opt_train(optimizer_fn, states, hyperparams, features, labels, save_path, batch_size=10, num_epochs=2):
    """
    A general train func for testing various optimizers and printing the trace of loss.
    The benchmark is a linear regression model with NASA data.
    """
    net, loss = linear_reg, squared_loss
    W = nn.Parameter(
        torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
        requires_grad=True
    )
    b = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        # evaluate the mean squared loss over the whole dataset
        return loss(net(features, W, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = tools.get_data_batch_torch(batch_size, features, labels)

    start = time.time()
    for _ in range(num_epochs):
        for batch_idx, (X, y) in enumerate(data_iter):
            l = loss(net(X, W, b), y).mean()
            if W.grad is not None:
                W.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            optimizer_fn([W, b], states, hyperparams)
            if (batch_idx + 1) * batch_size % 100 == 0:    # call once every 100 samples
                ls.append(eval_loss())

    print('loss: %f, %f sec per epoch' % (ls[-1], (time.time() - start) / num_epochs))
    tools.set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.show()


def opt_train_torch(optimizer_fn, optimizer_hyperparams, features, labels, save_path, batch_size=10, num_epochs=2):
    """
    A general train func for testing various optimizers and printing the trace of loss.
    The benchmark is a linear regression model with NASA data.
    Implement by Torch.
    """
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=True)

    start = time.time()
    for _ in range(num_epochs):
        for batch_idx, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y) / 2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if (batch_idx + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())

    print('loss: %f, %f sec per epoch' % (ls[-1], (time.time() - start) / num_epochs))
    tools.set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.show()


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
