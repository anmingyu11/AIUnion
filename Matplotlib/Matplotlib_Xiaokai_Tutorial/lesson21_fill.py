import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5 * np.pi, 1000)

y1 = np.sin(x)
y2 = np.sin(x * 2)

# plt.plot(x,y1)
# plt.plot(x,y2)
ax = plt.gca()
ax.plot(x, y1)
ax.plot(x, y2)
ax.fill_between(x, y1, y2, y2 > y1)
ax.fill_between(x, y1, y2, y2 < y1)
# plt.fill(x,y1,'b',alpha=.5)
# plt.fill(x,y2,'r',alpha=.5)
plt.show()
