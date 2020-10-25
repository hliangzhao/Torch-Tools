"""
Some utilities.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import random
import torch
import zipfile
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


def load_jay_lyrics(path='data/jaychou_lyrics.txt.zip'):
    """
    Load text dataset which is consists of Jay Chou's songs.
    :return:
        - corpus_indices: the list of indices of each char in the corpus in sequence
        - char_to_idx: the dict of (char, idx) for each unique char appeared in the corpus
        - idx_to_char: the list of unique chars in the corpus
        - vocab_size: the num of unique chars in the corpus
    """
    with zipfile.ZipFile(path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')[0:10000]

    # list of non-repeat chars
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def get_timeseries_data_batch_random(corpus_indices, batch_size, num_steps, device=None):
    """
    Get time series data in batch by random sampling.
    For each sample (x, y), x and y are both 1-dim tensor of size (num_steps).
    :param corpus_indices:
    :param batch_size: num of samples each minibatch includes
    :param num_steps: num of time steps each sample includes
    :param device:
    :return: X and Y are of size (batch_size, num_steps)
    """
    num_samples = (len(corpus_indices) - 1) // num_steps
    num_batches = num_samples // batch_size
    sample_indices = list(range(num_samples))
    random.shuffle(sample_indices)       # shuffle the order of all samples in the given corpus

    def _data(pos):
        """
        Get a sample of size num_steps since position pos.
        """
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(num_batches):
        i = i * batch_size
        batch_indices = sample_indices[i: i + batch_size]
        X, Y = [_data(j * num_steps) for j in batch_indices], [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def get_timeseries_data_batch_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    Get time series data in batch by consecutive sampling.
    Assume (x1, y1) and (x2, y2) in batch #1, (x3, y3) and (x4, y4) in batch #4. The the hidden state of x1[-1] can
    be used to initialized the hidden state of x3[0]; the hidden state of x2[-1] can be used to initialized the
    hidden state of x4[0].
    Tensor([[x1, y1],      --->     Tensor([[x3, y3],
            [x2, y2]])                      [x4, y4]])

    :param corpus_indices:
    :param batch_size: num of samples each minibatch includes
    :param num_steps: num of time steps each sample includes
    :param device:
    :return: X and Y are of size (batch_size, num_steps)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    num_chars = len(corpus_indices)
    # Retrieve one sample from each minibatch, batch_len is the upper bound of the num of chars in these samples
    batch_len = num_chars // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    num_batches = (batch_len - 1) // num_steps
    for i in range(num_batches):
        i = i * num_steps
        X, Y = indices[:, i: i + num_steps], indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def get_NASA_data():
    data = np.genfromtxt('data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), torch.tensor(data[:1500,  -1], dtype=torch.float32)


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

    # 7. test load_jay_lyrics
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_jay_lyrics()
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)

    # 8. test get_timeseries_data_batch
    my_seq = list(range(30))
    # for X, Y in get_timeseries_data_batch_random(my_seq, batch_size=2, num_steps=6):
    #     print('X: ', X, '\nY: ', Y, '\n')
    for X, Y in get_timeseries_data_batch_consecutive(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY: ', Y, '\n')
