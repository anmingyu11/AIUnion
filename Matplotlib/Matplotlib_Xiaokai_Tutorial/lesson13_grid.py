import numpy as np
import matplotlib.pyplot as plt


def plot_grid():
    x = np.linspace(1,10,100)
    y = x * 2
    plt.grid(True,color='b',linestyle='--',linewidth=.5)
    plt.plot(x,y)
    plt.show()

def plot_grid_oo():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(1,10,100)
    y = x * 2
    ax.grid(b=True,color='g')
    ax.plot(x,y)
    plt.show()

plot_grid_oo()
