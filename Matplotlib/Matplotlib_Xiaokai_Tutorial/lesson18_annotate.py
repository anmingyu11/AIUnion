import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 1)
y = x * x

plt.plot(x, y)
plt.annotate('goubi', xy=(0, 1), xytext=(0, 20),
             arrowprops=dict(facecolor='g',headlength=10,headwidth=30))
plt.show()
