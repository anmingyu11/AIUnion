import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sample():
    x = np.linspace(-10, 10, 100)
    y = x ** 2
    plt.plot(x, y)
    plt.show()


#plot_sample()

SHA = pd.read_csv('./data/000001.csv')

#print(df.dtypes)
SHA.Date = pd.to_datetime(SHA.Date)
# print(SHA.dtypes)
print(SHA.head())

from pandas.plotting import register_matplotlib_converters

def plot_date_open():
    plt.clf()
    #plt.plot(SHA.Date,SHA.Open)# 转时间戳
    plt.plot_date(SHA.Date,SHA.Open,fmt='-',color='red',marker='o',linestyle='--')
    plt.plot_date(SHA.Date,SHA.Close,fmt='-',color='green',marker='x',linestyle='--')
    plt.show()

plot_date_open()
