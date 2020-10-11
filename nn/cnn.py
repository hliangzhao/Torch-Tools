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


def batch_normalize(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    Batch normalization for input features.
    :param is_training: training mode or eval mode
    :param X: input feature
    :param gamma: the scale parameter
    :param beta: the shift parameter
    :param moving_mean: moving mean for inference
    :param moving_var: moving variance for inference
    :param eps: a small number adding on var
    :param momentum: the param for moving mean and moving var's update
    :return: the normalized out_features
    """
    if not is_training:
        # if in eval mode, use the saved moving_mean and moving_var
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # X should be a 2-dim tensor (batch_size, feature_num) [for linear layer]
        # or 4-dim tensor (batch_size, channel_num, h, w) [for conv layer]
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            # broadcast automatically
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # calculate the mean and var on the channel dim
            # keep X' shape!
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)

        # update moving_mean and moving_var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNormalizeLayer(nn.Module):
    def __init__(self, num_features, num_dims):
        """
        :param num_features: for conv layer, num_features is the channel_num; for linear layer, num_features is the unit num
        :param num_dims: 2 or 4
        """
        super(BatchNormalizeLayer, self).__init__()
        shape = None
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)

        # gamma is initialized as 1
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)
        # beta is initialized as 0
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            # move the cpu vars onto the device
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # self.training is False if called .eval()
        Y, self.moving_mean, self.moving_var = batch_normalize(self.training, X, self.gamma, self.beta,
                                                               self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


def BatchNormalizedLeNet():
    """
    LeNet-5 with batch normalization.
    """
    return nn.Sequential(
        nn.Conv2d(1, 6, 5),
        BatchNormalizeLayer(num_features=6, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5),
        BatchNormalizeLayer(num_features=16, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        metrics.FlattenLayer(),
        nn.Linear(16 * 4 * 4, 120),
        BatchNormalizeLayer(num_features=120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNormalizeLayer(num_features=84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )


def BatchNormalizedLeNet1():
    """
    LeNet-5 with batch normalization.
    """
    return nn.Sequential(
        nn.Conv2d(1, 6, 5),
        # num_dims is not required
        nn.BatchNorm2d(num_features=6),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(num_features=16),
        nn.Sigmoid(),
        nn.MaxPool2d(2),
        metrics.FlattenLayer(),
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm2d(num_features=120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm2d(num_features=84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )


class Residual(nn.Module):
    """
    A single residual module.
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
        The default Residual module in ResNet is similar to vgg_block.
        :param in_channels:
        :param out_channels:
        :param use_1x1conv: if you want to change channels num
        :param stride: (h, w) / stride is the out_h and out_w, respectively
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            # use 1x1 conv layer to change X's channels
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    """
    A residual block is a composition of residual modules.
    :param in_channels:
    :param out_channels:
    :param num_residuals: the number of residual modules
    :param first_block: if this is the first residual block, channels, h, and w do not change.
    Otherwise, channels x 2, h / 2, w / 2.
    :return:
    """
    if first_block:
        assert in_channels == out_channels
    blocks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # it is the first residual module of each residual block to change tensor size
            blocks.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blocks.append(Residual(out_channels, out_channels))
    # add * before blocks because we input it as a tuple (see model_construct.MySequential())
    return nn.Sequential(*blocks)


def ResNet18():
    # block 1 is similar to GooLeNet's (h, w) --> (h/2, w/2)
    # input tensor (batch_size, 1, 224, 224)
    # 1
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),     # (batch_size, 64, 112, 112)
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)          # (batch_size, 64, 56, 56)
    )
    # 4 x 2 x 2
    net.add_module('resnet_block_1', resnet_block(64, 64, 2, first_block=True))       # (batch_size, 64, 56, 56)
    net.add_module('resnet_block_2', resnet_block(64, 128, 2))                        # (batch_size, 128, 28, 28)
    net.add_module('resnet_block_3', resnet_block(128, 256, 2))                       # (batch_size, 128, 14, 14)
    net.add_module('resnet_block_4', resnet_block(256, 512, 2))                       # (batch_size, 128, 7, 7)
    # 1
    net.add_module('global_avg_pool', metrics.GlobalAvgPool2d())                      # (batch_size, 128, 1, 1)
    net.add_module('fc', nn.Sequential(metrics.FlattenLayer(), nn.Linear(512, 10)))   # (batch_size, 128) ---> (batch_size, 10)
    return net


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

    # test VGG-11
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

    # test batch normalization
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data')
    net = BatchNormalizedLeNet()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=5)
    print(BatchNormalizedLeNet1())

    # test ResNet-18
    train_iter, test_iter = tools.load_fashion_MNIST(batch_size=256, root='../data', resize=96)
    net = ResNet18()
    print(net)
    X = torch.rand(1, 1, 244, 244)
    for name, block in net.named_children():
        X = block(X)
        print(name, 'output shape: ', X.shape)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # metrics.cnn_train(net, train_iter, test_iter, optimizer, device, num_epochs=1)
