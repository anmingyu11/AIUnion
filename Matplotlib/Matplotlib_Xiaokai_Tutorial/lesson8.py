import numpy as np
import matplotlib.pyplot as plt


def plot_box_sample():
    data = np.random.normal(size=1000, loc=0, scale=1)
    plt.boxplot(
        data
        , sym='o'
        , whis=0.5
    )
    plt.show()


def plot_box_labels():
    label = ['A', 'B', 'C', 'D']
    data = np.random.normal(size=(1000,4), loc=0, scale=1)
    plt.boxplot(
        data
        , labels=label
        , sym='o'
        , whis=1
    )
    plt.show()

plot_box_labels()

# plot_box_sample()
