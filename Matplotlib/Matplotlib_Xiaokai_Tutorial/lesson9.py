import numpy as np
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def plot_rain_bow():
    x = np.arange(-10., 10., 0.05)
    y = -x ** 2
    for intercept, c in enumerate(colors):
        plt.plot(x, y+intercept*2,color=c)
    plt.show()

def plot_markers():
    pass


plot_rain_bow()