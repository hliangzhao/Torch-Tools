"""
Some utilities.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import random
import torch
import numpy as np
import torch.utils.data as D
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from collections import OrderedDict
import metrics


def get_data_batch(batch_size, features, labels):
    """
    Get data batches from given samples where features and labels are torch tensors.
    The first h_exp is sample_num, whatever dims each feature is.
    :return: data_iter consists of (X, y)
    """
    sample_num = len(features)
    indices = list(range(sample_num))
    random.shuffle(indices)
    for i in range(0, sample_num, batch_size):
        selected_indices = torch.tensor(indices[i: min(i + batch_size, sample_num)], dtype=torch.int64)
        yield torch.index_select(features, 0, selected_indices), torch.index_select(labels, 0, selected_indices)


def get_data_batch_torch(batch_size, features, labels):
    """
    Get data batches from given samples by torch.
    :return: data_iter consists of (X, y)
    """
    dataset = D.TensorDataset(features, labels)
    return D.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def load_fashion_MNIST(batch_size, resize=None, root='data', num_workers=4):
    """
    Load Fashion-MNIST dataset and return data in batch.
    """
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    trans.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
    # transform is the composition of operates: resize first, and then transform it to tensor
    transform = torchvision.transforms.Compose(trans)

    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    fashion_mnist_test = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    train_iter = D.DataLoader(
        fashion_mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = D.DataLoader(
        fashion_mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_iter, test_iter


def get_fashion_MNIST_labels(labels):
    """
    Get str labels from numbers 0-9 (labels).
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_fashion_MNIST(images, labels, save_path='figs/fashion_MNIST.png'):
    """
    Show multiple images and their labels in one line.
    """
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(10, 10))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.savefig(save_path)
    plt.show()


def xyplot(x_vals, y_vals, name, save_path):
    use_svg_display()
    set_figsize(figsize=(4, 2.5))
    # use tensor.detach().numpy() to get np array
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.savefig(save_path)
    plt.show()


def plot_funcs(func, name, save_path):
    """
    Plot [x, y1] and [x, y2] where y1 = f(x) and y2 = f'(x).
    """
    x = torch.arange(-8., 8., 0.1, requires_grad=True)
    y = func(x)
    xyplot(x, y, name, save_path)

    tmp = save_path.split('/')
    new_save_path = tmp[0] + '/' + tmp[1].split('.')[0] + '_grad.png'
    y.sum().backward()
    xyplot(x, x.grad, 'grad of ' + name, new_save_path)


def plot_semilogy(save_path, x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    use_svg_display()
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # 1. test get_data_batch and test get_data_batch_torch
    sample_num = 100
    in_feature = 2
    true_W, true_b = [2, -3.4], 4.2
    features = torch.randn(sample_num, in_feature, dtype=torch.float32)
    labels = true_W[0] * features[:, 0] + true_W[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    for X, y in get_data_batch(10, features, labels):
        print(X, '\n', y)
        break
    for X, y in get_data_batch_torch(10, features, labels):
        print(X, '\n', y)
        break

    # 2. test cross_entropy_loss
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5], [0.2, 0.2, 0.6]])
    y = torch.tensor([0, 2, 2], dtype=torch.int64)
    print(metrics.cross_entropy_loss(y_hat, y).sum())

    # 3. test softmax
    X = torch.rand((2, 5))
    print(metrics.softmax(X).sum(dim=1))

    # 4. test load_fashion_MNIST and show_fashion_MNIST
    train_iter, test_iter = load_fashion_MNIST(batch_size=10)
    images, labels = iter(train_iter).next()
    show_fashion_MNIST(images[0:10], get_fashion_MNIST_labels(labels[0:10]))

    # 5. test FlattenLayer and classify accuracy
    net = nn.Sequential(
        OrderedDict([
            ('flatten', metrics.FlattenLayer()),
            ('linear', nn.Linear(in_features=784, out_features=10))
        ])
    )
    print(metrics.evaluate_classify_accuracy(test_iter, net))

    # 6. test activate functions
    plot_funcs(nn.ReLU(), 'relu', save_path='figs/relu.png')
    plot_funcs(nn.Sigmoid(), 'sigmoid', save_path='figs/sigmoid.png')
    plot_funcs(nn.Tanh(), 'tanh', save_path='figs/tanh.png')
