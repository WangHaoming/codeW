# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 将父目录加入模块搜索路径
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # 加载 MNIST 测试集，像素值归一化到 [0,1]，图像展平为一维，标签为整数（非 one-hot）
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    # 从预训练权重文件中加载网络参数（W1/W2/W3 和 b1/b2/b3）
    # W1: 784x100, W2: 100x100, W3: 100x10  是隐藏层参数 输入 X Wi 可以得隐藏层状态  Bi 是偏置向量
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # 取出三层网络的权重矩阵和偏置向量
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 第一层：线性变换 + sigmoid 激活
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 第二层：线性变换 + sigmoid 激活
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 第三层（输出层）：线性变换 + softmax，输出各类别的概率分布
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 加载测试数据和网络参数
x, t = get_data()
network = init_network()

# 遍历测试集，收集预测错误的样本
errors = []
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)       # 取概率最大的类别作为预测结果
    if p != t[i]:          # 预测结果与真实标签不一致则记录为错误
        errors.append({
            'img': x[i].reshape(28, 28),  # 恢复为 28×28 的二维图像
            'true': t[i],                 # 真实标签
            'pred': p,                    # 预测标签
            'output': y                   # 输出层各类别概率
        })

print(f"总错误数: {len(errors)} / {len(x)}，错误率: {len(errors)/len(x)*100:.2f}%")

# 每页显示 10 张错误图片，支持翻页
page_size = 10
total = len(errors)
page = 0

while page * page_size < total:
    # 取当前页的错误样本切片
    batch = errors[page * page_size: (page + 1) * page_size]
    # 创建 2 行 × N 列的子图网格：第一行显示图片，第二行显示概率柱状图
    fig, axes = plt.subplots(2, len(batch), figsize=(len(batch) * 2, 5))
    fig.suptitle(f"预测错误样本 ({page*page_size+1}~{page*page_size+len(batch)} / {total})", fontsize=13)

    for col, item in enumerate(batch):
        # 上方：显示灰度手写图片，标题标注真实和预测标签
        ax_img = axes[0][col]
        ax_img.imshow(item['img'], cmap='gray')
        ax_img.set_title(f"真实:{item['true']} 预测:{item['pred']}", fontsize=9, color='red')
        ax_img.axis('off')

        # 下方：显示输出层各数字类别的概率柱状图
        # 红色=预测类别，绿色=真实类别，蓝色=其他类别
        ax_bar = axes[1][col]
        colors = ['red' if i == item['pred'] else ('green' if i == item['true'] else 'steelblue')
                  for i in range(10)]
        ax_bar.bar(range(10), item['output'], color=colors)
        ax_bar.set_xticks(range(10))
        ax_bar.set_ylim(0, 1)
        ax_bar.tick_params(labelsize=7)
        ax_bar.set_xlabel("数字类别", fontsize=7)

    plt.tight_layout()
    plt.show()

    page += 1
    # 还有下一页时，等待用户输入决定继续或退出
    if page * page_size < total:
        ans = input(f"按 Enter 查看下一页，输入 q 退出: ")
        if ans.strip().lower() == 'q':
            break
