# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
from traceback import print_tb
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    # X.ndim是什么？ 
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


def plot_function_surface(f, x_range=(-2, 2), y_range=(-2, 2), step=0.25):
    x0 = np.arange(x_range[0], x_range[1] + step, step)
    x1 = np.arange(y_range[0], y_range[1] + step, step)
    X, Y = np.meshgrid(x0, x1)
    # print(X)
    points = np.stack([X, Y], axis=1)
    print(points)
    Z = f(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="#4c566a", linewidth=1.2)
    ax.set_title("f(x0, x1) = x0^2 + x1^2")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("f(x)")
    plt.show()


if __name__ == '__main__':
    # 在三维坐标系上打印出所有点
    # plot_function_surface(function_2)
    # exit()


    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    # 生成 X Y 二维网络矩阵
    
    # 假设：

    #     x0 = [1, 2, 3]
    #     x1 = [10, 20]
    # 那么：

    #     X = [[1, 2, 3],
    #          [1, 2, 3]]

    #     Y = [[10, 10, 10],
    #          [20, 20, 20]]

    # 就是 X 按照 Y 的维度，分裂出多列，Y 按照 X 的维度分裂出多列

    X, Y = np.meshgrid(x0, x1)


    # 把矩阵展开成一维
    # X.flatten() -> [1, 2, 3, 1, 2, 3]
    # Y.flatten() -> [10, 10, 10, 20, 20, 20]
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")

    print(X)
    print( -grad[0])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
