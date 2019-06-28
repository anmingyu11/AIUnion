import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_no_related():
    N = 1000
    x = np.random.randn(N)
    y = np.random.randn(N)
    plt.clf()
    plt.scatter(x, y)
    plt.plot()
    plt.show()


# plot_no_related()

def plot_related(sign):
    N = 1000
    x = np.random.randn(N)
    y = np.sign(sign) * x + np.random.randn(N) * 0.5
    plt.clf()
    plt.scatter(x, y)
    plt.plot()
    plt.show()

# plot_related(-1)


def plot_SHA():
    ''' SHA  上证指数'''
    df = pd.read_csv('./data/000001.csv')
    df['change'] = df.Close - df.Open
    change = df['change']
    yesterday = change[:-1]
    today = change[1:]
    plt.clf()
    plt.scatter(yesterday,today,s=100,c='r',marker='<',alpha=0.5) # s指的是面积.
    plt.show()

plot_SHA()
