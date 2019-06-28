import numpy as np
import matplotlib.pyplot as plt


def plot_bar_example():
    N = 5
    y = [20, 10, 30, 25, 15]
    index = np.arange(N)
    # pl = plt.bar(x=index,height=y,width=0.5,color='r',align='horizontal')
    plt.clf()
    plt.barh(y=index, width=y, color='r')
    plt.show()


def plot_bar_two():
    N = 4
    index = np.arange(N)
    bar_width = 0.3
    sales_BJ = [52, 55, 63, 53]
    sales_SH = [44, 66, 55, 41]
    plt.clf()
    plt.bar(index, sales_BJ, bar_width, color='b')
    plt.bar(index + bar_width, sales_SH, bar_width, color='r')
    plt.show()


def plot_bar_cascading():
    n = 4
    index = np.arange(n)
    sales_bj = [52, 55, 63, 53]
    sales_sh = [44, 66, 55, 41]
    bar_width = 0.3
    plt.clf()
    plt.bar(index,sales_sh,bar_width)
    plt.bar(index,sales_bj,bar_width,bottom=sales_sh)
    plt.show()


plot_bar_cascading()
