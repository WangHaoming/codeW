# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # 随机设置默认参数权重
        # self.W = np.random.randn(2,3)
        # print(self.W)
        self.W = np.array([
            [ 0.76950497, -0.01518463, -1.94532802],
            [-1.37520222, -1.03283638, -0.36084079]
        ])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        print("loss is")
        print(loss)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

# 定义一个函数 def f(w)
#              return net.loss(x, t)  
# 函数中并没有使用 w
# 调用的时候传入   f(net.W)  -> return net.loss(x,t)

# f = lambda w: net.loss(x, t)

#  w 这个参数并没有意义，可以不传递， 在numerical_gradient中调用 f 的时候直接不传递参数也可以。
#  net其实也通过闭包传递给了 lambda 函数
f = lambda: net.loss(x, t)


dW = numerical_gradient(f, net.W)

print(dW)