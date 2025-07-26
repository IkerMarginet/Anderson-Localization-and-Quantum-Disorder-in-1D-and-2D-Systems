import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
from scipy.stats import gamma as gamma_dist

# --- Settings ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (12, 8),
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 13
})

# --- 1. Plot of the Gamma function ---
x = np.linspace(0.1, 5, 400)
y = gamma(x)

plt.figure()
plt.plot(x, y, label=r'$\Gamma(x)$', color='darkblue', lw=2)
plt.title("Gamma Function")
plt.xlabel(r'$x$')
plt.ylabel(r'$\Gamma(x)$')
plt.legend()
plt.grid(True)
plt.savefig("gamma_function_plot.png", dpi=300)
plt.close()

# --- 2. Comparison with factorial for integers ---
n = np.arange(1, 10)
gamma_vals = gamma(n)
fact_vals = factorial(n - 1, exact=True)

plt.figure()
plt.plot(n, gamma_vals, 'o-', label=r'$\Gamma(n)$', color='crimson')
plt.plot(n, fact_vals, 's--', label=r'$(n-1)!$', color='forestgreen')
plt.title("Gamma Function vs. Factorial")
plt.xlabel(r'$n$')
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("gamma_vs_factorial.png", dpi=300)
plt.close()

# --- 3. Application: Gamma distribution (used in statistics) ---
x = np.linspace(0, 20, 500)
shapes = [1, 2, 3, 5]

plt.figure()
for k in shapes:
    pdf = gamma_dist.pdf(x, a=k)
    plt.plot(x, pdf, label=fr'$k={k}$')

plt.title("Gamma Distribution (PDFs for different shapes)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig("gamma_distribution.png", dpi=300)
plt.close()

# --- 4. Gamma function in complex plane ---
from mpl_toolkits.mplot3d import Axes3D

# Create a meshgrid over complex plane
re = np.linspace(-3, 3, 300)
im = np.linspace(-3, 3, 300)
Re, Im = np.meshgrid(re, im)
Z = Re + 1j * Im
GZ = gamma(Z)

# Plot magnitude
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Re, Im, np.abs(GZ), cmap='viridis', rstride=5, cstride=5)
ax.set_title('Magnitude of $\Gamma(z)$ on Complex Plane')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_zlabel(r'$|\Gamma(z)|$')
plt.savefig("gamma_complex_plane.png", dpi=300)
plt.close()
