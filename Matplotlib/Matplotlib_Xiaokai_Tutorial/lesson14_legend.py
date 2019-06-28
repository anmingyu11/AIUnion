import matplotlib.pyplot as plt
import numpy as np

def plot_legend():
    x = np.arange(1, 11, 1)
    plt.clf()
    plt.plot(x, x * 3, label='Normal')
    plt.plot(x, x * 4, label='Fast')
    plt.plot(x, x * 5, label='Faster')
    plt.plot(x, x * 6, label='Faster1')
    plt.plot(x, x * 7, label='Faster2')
    plt.plot(x, x * 8, label='Faster3')
    plt.legend(loc='best', ncol=3)
    plt.show()

def plot_legend_oo():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, 11, 1)
    l = plt.plot(x,x,label='goubi')
    #ax.legend(['goubi'])
    ax.legend()
    plt.show()

plot_legend_oo()
