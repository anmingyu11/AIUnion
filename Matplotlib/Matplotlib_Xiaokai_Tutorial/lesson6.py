import numpy as np
import matplotlib.pyplot as plt

def plot_sample():
    mu = 100
    sigma = 20
    x = mu + sigma * np.random.randn(2000)

    plt.hist(x, bins=50, color='g', density=False)
    plt.show()

def plot_hist_2d():
    x = np.random.randn(1000) + 2
    y = np.random.randn(1000) + 3
    plt.hist2d(x,y,bins=40)
    plt.show()

plot_hist_2d()