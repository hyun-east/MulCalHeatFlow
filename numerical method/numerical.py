import numpy as np
import matplotlib.pyplot as plt

Nx, Ny = 1000, 600
Lx, Ly = 10.0, 5.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

H = np.random.normal(0, 1, (Nx, Ny))

for i in range(Nx):
    x = i * dx - 5.0
    if x < -1:
        H[i, 0] = 10
    elif -1 <= x < 0:
        H[i, 0] = 20
    elif 0 <= x < 1:
        H[i, 0] = 5
    else:
        H[i, 0] = -10

tolerance = 1e-5
max_iterations = 10000

for it in range(max_iterations):
    H_old = H.copy()


    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            H[i, j] = (H[i+1, j] + H[i-1, j]) / dx**2 + (H[i, j+1] + H[i, j-1]) / dy**2
            H[i, j] /= 2 * (1/dx**2 + 1/dy**2)

    for j in range(1, Ny - 1):
        H[0, j] = H[1, j]

    for j in range(1, Ny - 1):
        H[Nx - 1, j] = H[Nx - 2, j]

    for i in range(Nx):
        H[i, Ny - 1] = H[i, Ny - 2]

    print(np.linalg.norm(H - H_old))


    if np.linalg.norm(H - H_old) < tolerance:
        print(f'Converged after {it+1} iterations')
        break


x_vals = np.linspace(-Lx/2, Lx/2, Nx)
y_vals = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x_vals, y_vals)

plt.contourf(X, Y, H.T, 20, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution')
plt.show()