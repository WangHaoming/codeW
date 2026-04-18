# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        # print("target set:")
        # print(t)

        # target set:
        # [[0. 0. 0. ... 1. 0. 0.]
        #  [0. 0. 1. ... 0. 0. 0.]
        #  [0. 1. 0. ... 0. 0. 0.]


        y = self.predict(x)
        #  axis=0：按列方向
        #  axis=1：按行方向
        # 找到最大值的索引
        y = np.argmax(y, axis=1)
        # t 是 one hot 编码
        t = np.argmax(t, axis=1)        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # loss_W = lambda: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # 这个 gradient 方法是在做反向传播（backpropagation），也就是高效地计算两层神经网络里每个参数的梯度。
        # 它的作用和 two_layer_net.py 里的 numerical_gradient() 是一样的，都是求 W1, b1, W2, b2 的导数，但这个版本是“解析求导 + 链式法则”，
        # 比数值微分快很多。


        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        # x 是 100个验本
        batch_num = x.shape[0]
        
        # forward
        # 计算 y， 也就是计算结果
        a1 = np.dot(x, W1) + b1
        # 引入非线性，让网络不再是“直线模型”
        # 把结果转换成一个从 0 到 1 的数字，也就是概率
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        # dy 是误差
        dy = (y - t) / batch_num

        # dy 是最后一层导数，乘以z1.T就是计算出上一层的导出，根据反向传播原理
        grads['W2'] = np.dot(z1.T, dy)
        # 因为 b2 与 W 是加的关系，所以直接把 dy 反向传播给b2
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads