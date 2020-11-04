"""
This module visualizes how the optimization algorithms work.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import tools
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import metrics


features, labels = tools.get_NASA_data('../data/airfoil_self_noise.dat')


class ShowPoint:
    """
    Show the local minimum and saddle point.
    """
    @staticmethod
    def show_minimum():
        """
        Visualize the local and global minimum.
        """
        def f(x):
            return x * np.cos(np.pi * x)

        tools.use_svg_display()
        tools.set_figsize((4.5, 2.5))
        x = np.arange(-1., 2., 0.01)
        fig, = plt.plot(x, f(x))
        fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.), arrowprops=dict(arrowstyle='->'))
        fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.savefig('../figs/extreme_p.png')
        plt.show()

    @staticmethod
    def show_saddle():
        """
        Visualize the saddle point.
        """
        def f(x):
            return x**3

        tools.use_svg_display()
        tools.set_figsize((4.5, 2.5))
        x = np.arange(-2.0, 2.0, 0.1)
        fig, = plt.plot(x, f(x))
        fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0), arrowprops=dict(arrowstyle='->'))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.savefig('../figs/saddle_p.png')
        plt.show()

    @staticmethod
    def show_saddle_3d():
        """
        Visualize the 2-dim saddle point.
        """
        def f(x, y):
            return x ** 2 - y ** 2

        tools.use_svg_display()
        x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
        ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
        ax.plot_wireframe(x, y, f(x, y), **{'rstride': 2, 'cstride': 2})
        ax.plot([0], [0], [0], 'rx')
        ticks = [-1, 0, 1]
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('../figs/saddle2d_p.png')
        plt.show()


class ShowGD:
    """
    Use function f(x) = x^2 to show how the local minimum is found with gradient descent method.
    """
    @staticmethod
    def gd(eta):
        """
        Implement the gradient descent method.
        eta is the lr.
        """
        x = 10
        results = [x]
        for i in range(10):
            x -= eta * 2 * x
            results.append(x)
        print('epoch 10, x:', x)
        return results

    @staticmethod
    def show_trace(res):
        """
        Show the trace of update of local minimum.
        """
        n = max(abs(min(res)), abs(max(res)), 10)
        f_line = np.arange(-n, n, 0.01)
        tools.use_svg_display()
        tools.set_figsize()
        plt.plot(f_line, [x * x for x in f_line])
        plt.plot(res, [x * x for x in res], '-o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('../figs/gd_trace.png')
        plt.show()


def train_2d(trainer, eta):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2, eta)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (20, x1, x2))
    return results


def show_trace_2d(f, res, save_path):
    tools.use_svg_display()
    tools.set_figsize()
    plt.plot(*zip(*res), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1., 0.1), np.arange(-3., 1., 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)
    plt.show()


def f1_2d(x1, x2):
    """
    f(x, y) = x^2 + 2y^2.
    """
    return x1 ** 2 + 2 * x2 ** 2


def gd1_2d(x1, x2, s1, s2, eta=0.4):
    """
    Gradient descent for f(x, y) = x^2 + 2y^2.
    s1 and s2 are not used.
    """
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0


def sgd1_2d(x1, x2, s1, s2, eta=0.1):
    """
    Stochastic gradient descent for f(x, y) = x^2 + 2y^2.
    s1 and s2 are not used.
    """
    return x1 - eta * (2 * x1 + np.random.normal(0.1)), x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def f2_2d(x1, x2):
    """
    f(x, y) = 0.1 * x^2 + 2y^2.
    A test function for momentum.
    """
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd2_2d(x1, x2, s1, s2, eta=0.4):
    """
    Gradient descent for f(x, y) = 0.1 * x^2 + 2y^2.
    s1 and s2 are not used.

    If eta is set too small, the local optimal can not be achieved.
    If eta is set too large, the result will never convergence.
    """
    return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0


def momentum_2d(x1, x2, v1, v2, eta=0.4):
    """
    Momentum for f(x, y) = 0.1 * x^2 + 2y^2.
    """
    gamma = 0.6
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


def init_momentum_state():
    v_W = torch.zeros((features.shape[-1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return v_W, v_b


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data


def adagrad_2d(x1, x2, s1, s2, eta=0.4, eps=1e-6):
    """
    Adagrad for f(x, y) = 0.1 * x^2 + 2y^2.
    """
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def init_adagrad_state():
    s_W = torch.zeros((features.shape[-1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return s_W, s_b


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += p.grad.data ** 2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


def rmsprop_2d(x1, x2, s1, s2, eta=0.4, eps=1e-6, gamma=0.9):
    """
    RMSProp for f(x, y) = 0.1 * x^2 + 2y^2.
    """
    g1, g2 = 0.2 * x1, 4 * x2
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta * g1 / math.sqrt(s1 + eps)
    x2 -= eta * g2 / math.sqrt(s2 + eps)
    return x1, x2, s1, s2


def init_rmsprop_state():
    s_W = torch.zeros((features.shape[-1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return s_W, s_b


def rmsprop(params, states, hyperparams):
    eps, gamma = 1e-6, hyperparams['gamma']
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1 - gamma) * p.grad.data ** 2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


def test_sgd(lr, batch_size, save_path, num_epochs=2):
    metrics.opt_train(sgd, None, {'lr': lr}, features, labels, save_path, batch_size, num_epochs)


def test_sgd_torch(lr, batch_size, save_path, num_epochs=2):
    metrics.opt_train_torch(torch.optim.SGD, {'lr': lr}, features, labels, save_path, batch_size, num_epochs)


def init_adadelta_state():
    s_W, s_b = torch.zeros((features.shape[-1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_W, delta_b = torch.zeros((features.shape[-1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return (s_W, delta_W), (s_b, delta_b)


def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s = rho * s + (1 - rho) * (p.grad.data ** 2)
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta = rho * delta + (1 - rho) * g ** 2


def init_adam_states():
    v_W, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_W, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return (v_W, s_W), (v_b, s_b)


def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1


if __name__ == '__main__':
    # 1. show extreme point and saddle point
    ShowPoint.show_minimum()
    ShowPoint.show_saddle()
    ShowPoint.show_saddle_3d()

    # 2. show how the local minimum is found with gradient descent method under different lr
    ShowGD.show_trace(ShowGD.gd(0.2))
    ShowGD.show_trace(ShowGD.gd(0.5))
    ShowGD.show_trace(ShowGD.gd(1.1))

    # 3. test 2d (batch) gradient descent and stochastic gradient descent
    show_trace_2d(f1_2d, train_2d(gd1_2d, 0.3), save_path='../figs/gd_2d.png')
    show_trace_2d(f1_2d, train_2d(sgd1_2d, 0.1), save_path='../figs/sgd_2d.png')

    # 4. test bgd (1500 is the size of dataset)
    test_sgd(1, 1500, save_path='../figs/basic_batch_gd.png', num_epochs=6)
    # 5. test sgd
    test_sgd(0.005, 1, save_path='../figs/basic_sgd.png')
    # 6. test mini-batch gd
    test_sgd(0.05, 10, save_path='../figs/basic_minibatch_gd.png')
    # 7. test batch gd (torch)
    test_sgd_torch(0.05, 10, save_path='../figs/basic_minibatch_gd_torch.png')

    # 8. test why we need momentum
    show_trace_2d(f2_2d, train_2d(gd2_2d, 0.1), save_path='../figs/gd2_2d.png')
    show_trace_2d(f2_2d, train_2d(gd2_2d, 0.4), save_path='../figs/gd2_2d.png')
    show_trace_2d(f2_2d, train_2d(gd2_2d, 1.0), save_path='../figs/gd2_2d.png')

    # 9. test momentum
    show_trace_2d(f2_2d, train_2d(momentum_2d, 0.5), save_path='../figs/momentum_2d.png')
    metrics.opt_train(sgd_momentum, init_momentum_state(), {'lr': 0.02, 'momentum': 0.5}, features, labels,
                      '../figs/momentum1.png')
    metrics.opt_train(sgd_momentum, init_momentum_state(), {'lr': 0.02, 'momentum': 0.9}, features, labels,
                      '../figs/momentum2.png')
    metrics.opt_train(sgd_momentum, init_momentum_state(), {'lr': 0.004, 'momentum': 0.9}, features, labels,
                      '../figs/momentum3.png')
    metrics.opt_train_torch(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9}, features, labels,
                            '../figs/momentum4.png')

    # 10. test adagrad
    show_trace_2d(f2_2d, train_2d(adagrad_2d, 0.4), save_path='../figs/adagrad_2d.png')
    show_trace_2d(f2_2d, train_2d(adagrad_2d, 2), save_path='../figs/adagrad_2d.png')
    metrics.opt_train(adagrad, init_adagrad_state(), {'lr': 0.1}, features, labels, save_path='../figs/adagrad1.png')
    metrics.opt_train_torch(torch.optim.Adagrad, {'lr': 0.1}, features, labels, save_path='../figs/adagrad2.png')

    # 11. test RMSProp
    show_trace_2d(f2_2d, train_2d(rmsprop_2d, 0.4), save_path='../figs/rmsprop_2d.png')
    metrics.opt_train(adagrad, init_adagrad_state(), {'lr': 0.1, 'gamma': 0.9}, features, labels, save_path='../figs/rmsprop1.png')
    metrics.opt_train_torch(torch.optim.RMSprop, {'lr': 0.1, 'alpha': 0.9}, features, labels, save_path='../figs/rmsprop2.png')

    # 12. test AdaDelta
    metrics.opt_train(adadelta, init_adadelta_state(), {'rho': 0.99}, features, labels, save_path='../figs/adadelta1.png')
    metrics.opt_train_torch(torch.optim.Adadelta,  {'rho': 0.99}, features, labels, save_path='../figs/adadelta2.png')

    # 13. test adam
    metrics.opt_train(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels, save_path='../figs/adam1.png')
    metrics.opt_train_torch(torch.optim.Adam, {'lr': 0.01}, features, labels, save_path='../figs/adam2.png')
