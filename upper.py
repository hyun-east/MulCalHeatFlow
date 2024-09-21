from HeatMap import HeatMap
import numpy as np
import matplotlib.pyplot as plt

HM = HeatMap()
HM.c = {'c1': -10, 'c2' : 10, 'c3' : 20, 'c4': 5}
x = np.linspace(-5, 5, 1000)
y = np.linspace(1e-16, 5, 1000)
X, Y = np.meshgrid(x, y)

Z = HM.UpperH(X, Y)
Z2 = HM.UpperHConj(X, Y)

plt.contourf(X, Y, Z, levels = 20, cmap = 'coolwarm')
plt.contour(X, Y, Z2, levels = 20, cmap = 'grey')
plt.show()