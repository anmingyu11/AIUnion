import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 1)
y = x * x
plt.plot(x,y)
plt.text(-2,40,'fuc : y = x*x',family='serif',size=20,color='r')
plt.text(-2,30,'fuc : y = x*x',family='fantasy',size=20)
plt.show()
