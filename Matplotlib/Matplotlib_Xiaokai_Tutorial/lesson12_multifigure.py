import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure()

x = np.linspace(1,3,100)
y = x**2
ax1 = fig1.add_subplot(111)
ax1.plot(x,y)
plt.show()

fig2 = plt.figure()
x = np.linspace(1,3,100)
y = x**2
ax2 = fig2.add_subplot(111)
ax2.plot(x,y)
plt.show()
