import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 1)
y = x ** 2

plt.plot(x, y)
plt.axis([-10, 10, 1, 2])
plt.xlim(-10,10)
plt.ylim(0,1000)
plt.show()
