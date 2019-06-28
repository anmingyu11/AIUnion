import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([1, 7])
ax.set_ylim([1, 5])
ax.text(2, 4, r'$\alpha$')  # r 不转义.
plt.show()
