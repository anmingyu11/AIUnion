import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime

def plot_gca():
    x = np.arange(1, 11, 1)
    plt.plot(x, x)
    ax = plt.gca()
    ax.locator_params(nbins=10)
    plt.show()

#plot_gca()

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2016,1,1)
delta = datetime.timedelta(days=1)

dates = mpl.dates.drange(start,end,delta)

y = np.random.rand(len(dates))

fig = plt.figure()
ax = plt.gca()
ax.plot_date(dates,y,linestyle='-',marker='')
fig.autofmt_xdate()
plt.show()

