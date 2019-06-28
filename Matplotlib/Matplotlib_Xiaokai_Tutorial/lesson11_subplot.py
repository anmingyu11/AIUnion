'''
fig = plt.figure()
 figure实例
 axes实例

ax = fig.add_subplot(111)
- ax 实例
- 参数1,总行数
- 参数2,总列数
- 参数3,子图的位置.
- 在fig上添加axes
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 1)
y = x*2

def plot_subplot():
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.plot(x,y)
    ax2.plot(x,-y)
    ax3.plot(x,10 + y)
    plt.show()

plot_subplot()
