from HeatMap import HeatMap
import numpy as np
import matplotlib.pyplot as plt

HM = HeatMap()
HM.c = {'c1': 10, 'c2': 20, 'c3': 30, 'c4': 10}
u = v = np.linspace(-1, 1, 1000)
U, V = np.meshgrid(u, v)

Z = HM.DiskH(U, V)
Z2 = HM.DiskHConj(U, V)

contour = plt.contourf(U, V, Z, levels = 200, cmap = 'coolwarm')
plt.contour(U, V, Z2, levels = 50, colors='black',  linestyles='solid')
plt.axis('equal')
plt.colorbar(contour, label='Temperature')
plt.show()