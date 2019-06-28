# pyplot
# pyplab
# 面向对象
from pylab import *
import matplotlib.pyplot as plt
import numpy as np


def pyplab_sample():
    x = arange(0, 10, 1)
    y = randn(len(x))
    plot(x, y)
    title('pylab')
    show()


# pyplab_sample()

def pyplot_sample():
    x = arange(0, 10, 1)
    y = randn(len(x))
    plt.plot(x, y)
    plt.title('pyplot')
    plt.show()

#pyplot_sample()

def plt_oo_sample():
    x = arange(0, 10, 1)
    y = randn(len(x))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l,=plt.plot(x,y)
    t = ax.set_title('oo')
    plt.show()

plt_oo_sample()
