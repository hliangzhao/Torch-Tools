import random
import torch
import numpy as np
import torch.utils.data as D
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from collections import OrderedDict


def get_data_batch(batch_size, features, labels):
    """
    Get data batches from given samples. Each sample is consists of (features, label).
    features and labels are torch tensors.
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
    """
    dataset = D.TensorDataset(features, labels)
    return D.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def squared_loss(y_hat, y):
    """
    y_hat and y are 1-dim torch tensor with size (sample_num).
    What is returned is a torch tensor with the same size as y_hat's (y's).
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def cross_entropy_loss(y_hat, y):
    """
    y_hat is of size (sample_num, label_num), y is of size (sample_num), where each element indicates the
    true label of each sample.
    y.view(-1, 1) makes y being of new size: (sample_num, 1).
    """
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def get_classify_accuracy(y_hat, y):
    """
    Get accuracy for classification task.
    Get the accuracy of samples, where y_hat is of size (sample_num, label_num), y is of size (sample_num).
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_classify_accuracy(data_iter, model, W, b):
    """
    Calculate the accuracy over the data_iter in batch way for classification task.
    """
    acc_sum, n = 0., 0
    for X, y in data_iter:
        acc_sum += (model(X, W, b).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def evaluate_classify_accuracy_net(data_iter, net):
    """
    Calculate the accuracy over the data_iter in batch way for classification task.
    Use for model defined by torch.
    """
    acc_sum, n = 0., 0
    for X, y in data_iter:
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


def load_fashion_MNIST(batch_size, resize=None, root='data', num_workers=4):
    """
    Load Fashion-MNIST dataset and return data in batch.
    """
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
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


class FlattenLayer(nn.Module):
    """
    Flatten the img (2-dim or 3-dim tensor) into a vector.
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


def universal_train(train_iter, test_iter, net, loss, optimizer, epoch_num):
    """
    A universal training function for network defined by torch and used for classification.
    """
    for epoch in range(epoch_num):
        train_ls_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = net(X)
            ls = loss(y_hat, y).sum()
            optimizer.zero_grad()

            # call backward() to calculate the grad of params
            ls.backward()
            # use optimizer to update the grad for one step
            optimizer.step()

            train_ls_sum += ls.item()
            train_acc_sum += get_classify_accuracy(y_hat, y)
            n += y.shape[0]

        # evaluate accuracy on the test set
        test_acc = evaluate_classify_accuracy_net(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_ls_sum / n, train_acc_sum / n, test_acc))


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
    print(cross_entropy_loss(y_hat, y).sum())

    # 3. test softmax
    X = torch.rand((2, 5))
    print(softmax(X).sum(dim=1))

    # 4. test load_fashion_MNIST and show_fashion_MNIST
    train_iter, test_iter = load_fashion_MNIST(batch_size=10)
    images, labels = iter(train_iter).next()
    show_fashion_MNIST(images[0:10], get_fashion_MNIST_labels(labels[0:10]))

    # 5. test FlattenLayer and classify accuracy
    net = nn.Sequential(
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(in_features=784, out_features=10))
        ])
    )
    print(evaluate_classify_accuracy_net(test_iter, net))
