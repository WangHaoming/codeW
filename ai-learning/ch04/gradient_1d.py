# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    # 计算函数在x处的导数，使用中心差分法，计算结果是斜率，也就是切线斜率
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 

# 计算切线方程，返回一个函数，返回一条直线
# f: 是目标函数 x: 是切点处的x坐标
def tangent_line(f, x):
    # 计算切点处的斜率，根据目标函数
    d = numerical_diff(f, x)
    print(d)
    # 计算切线方程的截距，根据切点处的函数值和斜率
    y = f(x) - d*x
    # 切线方程： 在得到斜率之后，只要给出一个点的坐标，就可以计算出整个斜线
    
    # 切点一定在切线上，所以在切点处计算出来的y 值也是 f(x)的值。
    # 斜线（切线）方程：y = kx + b -> b = y - kx , 上面的 y 实际上切线方程中的截距 b， 下面的 t 是函数参数： y = kx + b 中的 x
    # d是上面计算得到的写斜率 k， 所以 d*t 就是相当于kx +y 就相当于斜线方程中的 +b
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

# tf = tangent_line(function_1, 5)
# y2 = tf(x)
print(x)
print(y)
# todo: 使用plot绘制一条直线，斜率是 numerical_diff(f, x)

x0 = 10
slope = numerical_diff(function_1, x0)   # 切点处的斜率
y0 = function_1(x0)    
# y_tangent的结果是一个向量
# 推导公式：y - y0 = slope * (x - x0) -> y = slope(x - x0) + y0
y_tangent = slope * (x - x0) + y0

# 绘制切线
plt.plot(x, y)
plt.plot(x, y_tangent, label="tangent")    
# plt.plot(x, y2)
plt.show()