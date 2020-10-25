"""
This module visualizes how the optimization algorithms work.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import tools
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_minimum():
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


def show_saddle():
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


def show_saddle_3d():
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


def gd(eta):
    """
    Gradient descent for function f(x) = x^2. eta is the lr.
    """
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results


def show_trace(res):
    """
    Show the trace of extreme point's finding for f(x) = x^2.
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


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
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


def f_2d(x1, x2):
    """
    f(x, y) = x^2 + 2y^2.
    """
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2, eta=0.1):
    """
    Gradient descent for f(x, y) = x^2 + 2y^2.
    s1 and s2 are not used.
    """
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0


def sgd_2d(x1, x2, s1, s2, eta=0.1):
    return x1 - eta * (2 * x1 + np.random.normal(0.1)), \
           x2 - eta * (4 * x2 + np.random.normal(0.1)), \
           0, 0


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


if __name__ == '__main__':
    # 1. show extreme point and saddle point
    show_minimum()
    show_saddle()
    # show_saddle_3d()

    # 2. test lr of gradient descent
    show_trace(gd(0.2))
    show_trace(gd(0.5))
    show_trace(gd(1.1))

    # 3. test 2d (stochastic) gradient descent
    show_trace_2d(f_2d, train_2d(gd_2d), save_path='../figs/gd_2d.png')
    show_trace_2d(f_2d, train_2d(sgd_2d), save_path='../figs/sgd_2d.png')
