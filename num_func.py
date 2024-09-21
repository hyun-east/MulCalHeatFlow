import matplotlib.pyplot as plt
import numpy as np


class NumericalHeatMap:
    def __init__(self):
        self.Nx, self.Ny = 800, 200
        self.Lx, self.Ly = 60.0, 5.0
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.tolerance = 1e-3
        self.max_iterations = 10000
        self.boundary_list = [-1, 0, 1]
        self.boundary_value = [10, 20,5, -10]

        self.H = None

    def ConformalMapping(self, u, v):
        denominator = u ** 2 + (1 - v) ** 2
        x = (2 * u) / denominator
        y = (1 - u ** 2 - v ** 2) / denominator
        return x, y

    def CalculateNumericalUpperPlane(self):
        boundary_list = self.boundary_list
        boundary_value = self.boundary_value

        self.H = np.zeros((self.Nx, self.Ny))

        for i in range(self.Nx):
            x = i * self.dx - self.Lx / 2
            if x < boundary_list[0]:
                self.H[i, 0] = boundary_value[0]
            for j in range(len(boundary_list) - 1):
                if boundary_list[j] <= x < boundary_list[j + 1]:
                    self.H[i, 0] = boundary_value[j + 1]
            if x >= boundary_list[-1]:
                self.H[i, 0] = boundary_value[-1]

        for it in range(self.max_iterations):
            H_old = self.H.copy()

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    self.H[i, j] = (self.H[i + 1, j] + self.H[i - 1, j]) / self.dx ** 2 + \
                                   (self.H[i, j + 1] + self.H[i, j - 1]) / self.dy ** 2
                    self.H[i, j] /= 2 * (1 / self.dx ** 2 + 1 / self.dy ** 2)

            print(np.linalg.norm(self.H - H_old))
            if np.linalg.norm(self.H - H_old) < self.tolerance:
                print(f'Converged after {it + 1} iterations')
                break

    def UpperH(self, x, y):
        if self.H is None:
            raise ValueError("Numerical values have not been calculated. Call CalculateNumericalUpper() first.")

        ix = int((x + self.Lx / 2) / self.dx)
        iy = int(y / self.dy)

        ix = min(max(0, ix), self.Nx - 1)
        iy = min(max(0, iy), self.Ny - 1)

        return self.H[ix, iy]

    def GetNumericalValue(self, X, Y):
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.UpperH(X[i, j], Y[i, j])
        return Z

    def DiskH(self, u, v):
        x, y = self.ConformalMapping(u, v)
        ret = np.vectorize(self.UpperH)(x, y)
        ret[u**2 + v**2 > 1] = np.nan
        return ret


HM = NumericalHeatMap()

HM.CalculateNumericalUpperPlane()
'''
x = np.linspace(-5, 5, 100)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)
Z = HM.GetNumericalValue(X, Y)

plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
plt.colorbar()
plt.show()'''

u = v = np.linspace(-1, 1, 1000)
U, V = np.meshgrid(u, v)

Z = HM.DiskH(U, V)

contour = plt.contourf(U, V, Z, levels = 200, cmap = 'coolwarm')
plt.axis('equal')
plt.colorbar(contour, label='Temperature')
plt.show()