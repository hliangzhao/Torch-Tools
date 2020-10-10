"""
This module defines popular CNN models.
"""
import torch
import tools
import metrics
from torch import nn, optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet5(nn.Module):
    """
    This LeNet-5 is used to classify Fashion_MNIST dataset.
    Pay attention to the first linear layer's input feature size.
    """
    def __init__(self):
        # input feature is (batch_size, in_channel, f_h, f_w) = (batch_size, 1, 28, 28)
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),     # (batch_size, 6, 24, 24)
            nn.Sigmoid(),
            nn.MaxPool2d(2),        # (batch_size, 6, 12, 12)

            nn.Conv2d(6, 16, 5),    # (batch_size, 16, 8, 8)
            nn.Sigmoid(),
            nn.MaxPool2d(2)         # (batch_size, 16, 4, 4)
        )
        # feature.view() is necessary !!!
        # actually this is what FlattenLayer do
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),     # (batch_size, 256)
            nn.Sigmoid(),
            nn.Linear(120, 84),             # (batch_size, 84)
            nn.Sigmoid(),
            nn.Linear(84, 10)               # (batch_size, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1))


class AlexNet(nn.Module):
    """
    This AlexNet is used to classify Fashion_MNIST dataset.
    The images in the dataset are resize into (1, 224, 224).
    Pay attention to the first linear layer's input feature size.
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        # input feature is of size (batch_size, 1, 224, 224)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),            # (batch_size, 96, 54, 54)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),                 # (batch_size, 96, 26, 26)

            nn.Conv2d(96, 256, 5, 1, 2),        # (batch_size, 256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),                 # (batch_size, 256, 11, 11)

            nn.Conv2d(256, 384, 3, 1, 1),       # (batch_size, 384, 11, 11)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),       # (batch_size, 384, 11, 11)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),       # (batch_size, 256, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(3, 2)                  # (batch_size, 256, 5, 5)
        )
        # feature.view() is necessary !!!
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),       # (batch_size, 256 * 5 * 5)
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1))


def vgg_block(num_convs, in_channels, out_channels):
    blocks = []
    for i in range(num_convs):
        if i == 0:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blocks.append(nn.ReLU())
    blocks.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*blocks)


def VGG11(conv_arch, fc_features, fc_hidden_units=4096, fc_out_units=10):
    """
    VGG-11 for Fashion MNIST dataset classification.
    :param conv_arch: tuple of input params for vgg_block()
    :param fc_features: the input feature number of the first linear layer
    :param fc_out_units: the number of output of the net
    :param fc_hidden_units: the number of hidden units
    :return: the constructed VGG-11 net
    """
    net = nn.Sequential()
    for i, (num_conv, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i + 1), vgg_block(num_conv, in_channels, out_channels))
    net.add_module('fc', nn.Sequential(
        metrics.FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_out_units)
    ))
    return net


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    """
    A network-in-network block is a conv_layer appended with two 1x1 conv_layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


def NiN():
    """
    NiN has the same feature change as AlexNet.
    """
    # input feature is of size (batch_size, 1, 224, 224)
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(3, 2),

        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(3, 2),

        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(3, 2),

        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),   # (batch_size, 10, 5, 5)
        metrics.GlobalAvgPool2d(),                                # (batch_size, 10, 1, 1)
        metrics.FlattenLayer()                                    # (batch_size, 10)
    )


class Inception(nn.Module):
    """
    The Inception module for GoogLeNet.
    Inception module does not change feature's height and width.
    """
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        assert p1.shape[2:] == p2.shape[2:] == p3.shape[2:] == p4.shape[2:]
        return torch.cat((p1, p2, p3, p4), dim=1)   # cat at the channel dim


def GoogLeNet():
    """
    The GoogLeNet for Fashion MNIST dataset classification.
    """
    # assume input tensor (batch_size, 1, 96, 96)
    return nn.Sequential(
        # block 1
        nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),    # (batch_size, 64, 48, 48)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)         # (batch_size, 64, 24, 24)
        ),
        # block 2
        nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),                        # (batch_size, 64, 24, 24)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),            # (batch_size, 192, 24, 24)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)         # (batch_size, 192, 12, 12)
        ),
        # block 3
        nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),             # out_channels = 256
            Inception(256, 128, (128, 192), (32, 96), 64),           # out_channels = 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)         # (batch_size, 192, 6, 6)
        ),
        # block 4
        nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)         # (batch_size, 832, 3, 3)
        ),
        # block 5
        nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),         # (batch_size, 1024, 3, 3)
            metrics.GlobalAvgPool2d()                                # (batch_size, 1024, 1, 1)
        ),
        metrics.FlattenLayer(),                                      # (batch_size, 1024)
        nn.Linear(1024, 10)                                          # (batch_size, 10)
    )


if __name__ == '__main__':
    # test LeNet-5
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data')
    net = LeNet5()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=5)

    # test AlexNet
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data', resize=224)
    net = AlexNet()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=1)

    # test vgg
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data', resize=224)
    conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    # 224 / 2^5 = 7
    fc_features = 512 * 7 * 7
    net = VGG11(conv_arch, fc_features)
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, block in net.named_children():
        X = block(X)
        print(name, 'output shape: ', X.shape)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=1)

    # test NiN
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data', resize=224)
    net = NiN()
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, block in net.named_children():
        X = block(X)
        print(name, 'output shape: ', X.shape)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=1)

    # test GooLeNet
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data', resize=96)
    net = GoogLeNet()
    print(net)
    X = torch.rand(1, 1, 96, 96)
    for name, block in net.named_children():
        X = block(X)
        print(name, 'output shape: ', X.shape)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=1)

