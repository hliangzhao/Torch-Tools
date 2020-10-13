"""
This module lists typical methods for constructing deep neuron networks.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
from torch import nn
from collections import OrderedDict


# standard construction
class MLP2h(nn.Module):
    def __init__(self, num_inputs, num_hiddens1, num_hiddens2, num_outputs):
        super(MLP2h, self).__init__()
        self.hidden1 = nn.Linear(num_inputs, num_hiddens1)
        self.act1 = nn.ReLU()
        self.hiddens2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(num_hiddens2, num_outputs)

    def forward(self, X):
        h1 = self.act1(self.hidden1(X))
        h2 = self.act2(self.hiddens2(h1))
        return self.output(h2)


# construct by nn.Sequential
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X


# ModuleList (no order, no correlated modules, no forward implemented)
# ModuleList is not the same as list
class FlexibleNet(nn.Module):
    def __init__(self):
        super(FlexibleNet, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        output = self.linear(x)
        for i, l in enumerate(self.linears):
            output = self.linears[i // 2](output) + l(output)
        return output


# use Module to construct a fancy net (a net with fancy forwarding computation)
class FancyNet(nn.Module):
    def __init__(self):
        super(FancyNet, self).__init__()
        self.constants = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        output = self.linear(X)
        output = nn.functional.relu(torch.mm(output, self.constants) + 1)
        output = self.linear(output)
        while output.norm().item() > 1:
            output /= 2
        if output.norm().item() < 0.8:
            output *= 10
        return output.sum()


# net without parameters
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, X):
        return X - X.mean()


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(20, 20), nn.ReLU())

    def forward(self, X):
        return self.net(X)


# DIY a net with self-defined parameters (use ParameterList)
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        # params should be of type nn.Parameter
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4), requires_grad=True) for _ in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1), requires_grad=True))

    def forward(self, X):
        for i in range(len(self.params)):
            X = torch.mm(X, self.params[i])
        return X


# DIY a net with self-defined parameters (use ParameterDict)
class MyDenseDict(nn.Module):
    def __init__(self):
        super(MyDenseDict, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4), requires_grad=True),
            'linear2': nn.Parameter(torch.randn(4, 1), requires_grad=True)
        })
        self.params.update({'linear3': nn.Parameter(torch.rand(4, 2), requires_grad=True)})

    def forward(self, X, choose='linear1'):
        return torch.mm(X, self.params[choose])


if __name__ == '__main__':
    # test MLP2h
    X = torch.rand(2, 784)
    net = MLP2h(784, 256, 64, 10)
    print(net)
    print(net(X))

    # test MySequential 1
    net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    print(net)
    print(net(X))

    # test MySequential 2
    modules = OrderedDict()
    modules['linear1'] = nn.Linear(784, 256)
    modules['act1'] = nn.ReLU()
    modules['linear2'] = nn.Linear(256, 64)
    modules['act2'] = nn.ReLU()
    modules['softmax'] = nn.Linear(64, 10)
    net = MySequential(modules)
    print(net)
    print(net(X))

    # test ModuleList 1
    net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
    net.append(nn.Linear(256, 64))
    net.append(nn.ReLU())
    net.append(nn.Linear(64, 10))
    print(net)
    try:
        print(net(X))
    except NotImplementedError:
        print('forward is not implemented')

    # test ModuleList 2
    net = FlexibleNet()
    print(net)
    print(net(X))

    # test ModuleDict
    net = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'action': nn.ReLU(),
    })
    net['output'] = nn.Linear(256, 10)
    print(net)
    try:
        print(net(X))
    except BaseException:
        print('forward is not implemented')

    # test FancyNet
    X = torch.rand(2, 20)
    net = FancyNet()
    print(net)
    print(net(X))

    # test complex net
    net = nn.Sequential(NestMLP(), nn.Linear(20, 20), FancyNet())
    print(net)
    print(net(X))

    # test CenteredLayer
    layer = CenteredLayer()
    print(layer(torch.tensor([1, 2, 3, 4], dtype=torch.float)))
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    y = net(torch.rand(4, 8))

    # test MyDense
    net = MyDense()
    print(net)
    print(net(torch.rand(1, 4)))

    # test MyDenseDict
    net = MyDenseDict()
    print(net)
    X = torch.ones(1, 4)
    print(net(X, 'linear1'))
    print(net(X, 'linear2'))
    print(net(X, 'linear3'))

    net = nn.Sequential(
        MyDenseDict(),
        MyDense(),
    )
    print(net)
    print(net(X))

