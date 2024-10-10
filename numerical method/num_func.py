import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time
from scipy.interpolate import griddata

class NumericalHeatMap:
    def __init__(self):
        self.Nx, self.Ny = 800, 400
        self.Lx, self.Ly = 40.0, 2.0
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.tolerance = 2e-5
        self.max_iterations = 2e5
        self.boundary_list = [-1, 0, 1]
        self.boundary_value = [10, 20, 5, -10]
        self.H = None

    def ConformalMapping(self, u, v):
        denominator = u ** 2 + (1 - v) ** 2
        x = (2 * u) / denominator
        y = (1 - u ** 2 - v ** 2) / denominator
        return x, y

    @jit(nopython=True)
    def solve_laplace(H, Nx, Ny, dx, dy, tolerance, max_iterations):
        for it in range(max_iterations):
            H_old = H.copy()

            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    H[i, j] = (H[i + 1, j] + H[i - 1, j]) / dx ** 2 + \
                              (H[i, j + 1] + H[i, j - 1]) / dy ** 2
                    H[i, j] /= 2 * (1 / dx ** 2 + 1 / dy ** 2)

            for j in range(1, Ny - 1):
                H[0, j] = H[1, j]

            for j in range(1, Ny - 1):
                H[Nx - 1, j] = H[Nx - 2, j]

            for i in range(Nx):
                H[i, Ny - 1] = H[i, Ny - 2]

            difference = np.linalg.norm(H - H_old)
            print("Iteration:", it, "Difference:", difference)

            if difference < tolerance:
                print(f'Converged after {it + 1} iterations')
                break

        return H

    def CalculateNumericalUpperPlane(self):
        boundary_list = self.boundary_list
        boundary_value = self.boundary_value

        self.H = np.random.normal(min(self.boundary_value), max(self.boundary_value), (self.Nx, self.Ny))

        for i in range(self.Nx):
            x = i * self.dx - self.Lx / 2
            if x < boundary_list[0]:
                self.H[i, 0] = boundary_value[0]
            for j in range(len(boundary_list) - 1):
                if boundary_list[j] <= x < boundary_list[j + 1]:
                    self.H[i, 0] = boundary_value[j + 1]
            if x >= boundary_list[-1]:
                self.H[i, 0] = boundary_value[-1]

        self.H = NumericalHeatMap.solve_laplace(self.H, self.Nx, self.Ny, self.dx, self.dy, self.tolerance, self.max_iterations)

    def UpperH(self, x, y):
        if self.H is None:
            raise ValueError("Numerical values have not been calculated. Call CalculateNumericalUpperPlane() first.")

        ix = int((x + self.Lx / 2) / self.dx)
        iy = int(y / self.dy)

        ix = min(max(0, ix), self.Nx - 1)
        iy = min(max(0, iy), self.Ny - 1)

        return self.H[ix, iy]

    def UpperH_2(self, X, Y):
        return np.vectorize(self.UpperH)(X, Y)

    def DiskH(self, u, v):
        x, y = self.ConformalMapping(u, v)
        ret = np.vectorize(self.UpperH)(x, y)
        ret[u**2 + v**2 > 1] = np.nan
        return ret

# 인스턴스 생성
HM = NumericalHeatMap()

boundary_list = [0, 45, 90, 180, 225, 270, 300]
boundary_value = [10, 20, 20, -10, 20, 10, 30]
temp_max = -1
for i in range(len(boundary_list)):
    if 0 <= boundary_list[0] < 90:
        boundary_list = boundary_list[1:] + boundary_list[:1]
        boundary_value = boundary_value[1:] + boundary_value[:1]
        temp_max = i
boundary_list.remove(90)

for i in range(len(boundary_list)):
    boundary_list[i] = HM.ConformalMapping(np.cos(np.radians(boundary_list[i])), np.sin(np.radians(boundary_list[i])))[0]

HM.Lx = max(6 * max(np.vectorize(abs)(boundary_list)), 40)
HM.boundary_list = boundary_list
HM.boundary_value = boundary_value

t = time.time()

HM.CalculateNumericalUpperPlane()

print(time.time() - t, "s")

x = np.linspace(-20, 20, 400)
y = np.linspace(-0.1, 20, 400)
X, Y = np.meshgrid(x, y)
Z = HM.UpperH_2(X, Y)

interp_x = np.linspace(-20, 20, 1600)
interp_y = np.linspace(-0.1, 20, 1600)
interp_X, interp_Y = np.meshgrid(interp_x, interp_y)

Z_interp = griddata((X.flatten(), Y.flatten()), Z.flatten(), (interp_X, interp_Y), method='cubic')

dZdx, dZdy = np.gradient(Z, x, y)
flow_magnitude = np.sqrt(dZdx**2 + dZdy**2)

flow_magnitude_interp = griddata((X.flatten(), Y.flatten()), flow_magnitude.flatten(),
                                 (interp_X, interp_Y), method='cubic')
u = v = np.linspace(-1.1, 1.1, 1000)
U, V = np.meshgrid(u, v)

mapped_X, mapped_Y = HM.ConformalMapping(U, V)

x_min, x_max = -HM.Lx / 2, HM.Lx / 2
y_min, y_max = 0, HM.Ly

mapped_X = np.clip(mapped_X, x_min, x_max)
mapped_Y = np.clip(mapped_Y, y_min, y_max)

Z2_interp = griddata((interp_X.flatten(), interp_Y.flatten()), Z_interp.flatten(),
                     (mapped_X.flatten(), mapped_Y.flatten()), method='cubic')

flow_magnitude_mapped = griddata((interp_X.flatten(), interp_Y.flatten()), flow_magnitude_interp.flatten(),
                                 (mapped_X.flatten(), mapped_Y.flatten()), method='cubic')

Z2_interp = Z2_interp.reshape(U.shape)
Z2_interp[U**2 + V**2 > 1] = np.nan
flow_magnitude_mapped = flow_magnitude_mapped.reshape(U.shape)
flow_magnitude_mapped[U**2 + V**2 > 1] = np.nan

plt.figure(figsize=(12, 6))

plt.subplot(121)
contourf = plt.contourf(interp_X, interp_Y, Z_interp, levels=200, cmap='coolwarm')
contours = plt.contour(interp_X, interp_Y, Z_interp, levels=100, colors='black')
plt.contour(interp_X, interp_Y, flow_magnitude_interp, levels=20, colors='gray')
plt.xlim(-3, 3)
plt.ylim(0, 2)
plt.clabel(contours, inline=True, fontsize=8)

plt.subplot(122)
contourf_disk = plt.contourf(U, V, Z2_interp, levels=200, cmap='coolwarm')
contours_disk = plt.contour(U, V, Z2_interp, levels=20, colors='black')
plt.contour(U, V, flow_magnitude_mapped, levels=20, colors='gray')
plt.axis('equal')
plt.clabel(contours_disk, inline=True, fontsize=8)

plt.show()
