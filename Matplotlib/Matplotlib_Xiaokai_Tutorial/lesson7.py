import matplotlib.pyplot as plt
import numpy as np


def plot_pie_sample():
    labels = ['A', 'B', 'C', 'D']
    fracs = [15, 30, 45, 10]
    plt.axes(aspect=1)
    explode = np.full(shape=4, fill_value=.05)
    explode[2] = 0.5
    plt.pie(x=fracs, labels=labels, autopct='%.0f%%', explode=explode, shadow=True)
    plt.show()


plot_pie_sample()
