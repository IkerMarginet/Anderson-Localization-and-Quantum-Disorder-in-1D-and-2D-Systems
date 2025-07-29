import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import sys

# Enable LaTeX rendering for high-quality plots (disable if LaTeX is not installed)
plt.rcParams['text.usetex'] = True  # Set to False if LaTeX causes errors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Parameters
t = 1.0  # Hopping amplitude
N = 500  # Number of sites (moderate for dense matrix efficiency)
W_vals = [0.5, 2.0, 5.0]  # Disorder strengths
num_realizations = 20  # Number of disorder realizations (more for smoother histograms)
num_bins = 50  # Number of histogram bins

def build_hamiltonian(N, W, t):
    """
    Construct the 1D Anderson model Hamiltonian as a sparse tridiagonal matrix.
    Args:
        N: Number of sites
        W: Disorder strength
        t: Hopping amplitude
    Returns:
        H: Dense Hamiltonian matrix
    """
    eps = np.random.uniform(-W/2, W/2, N)  # Random on-site potentials
    diagonals = [eps, -t*np.ones(N-1), -t*np.ones(N-1)]  # Diagonal and off-diagonal terms
    offsets = [0, 1, -1]  # Positions of diagonals
    return diags(diagonals, offsets, shape=(N, N)).toarray()

def level_statistics(evals):
    """
    Compute normalized level spacings for eigenvalue statistics.
    Args:
        evals: Eigenvalues
    Returns:
        s: Normalized spacings (s_i = (E_{i+1} - E_i) / mean_spacing)
    """
    evals = np.sort(evals)
    spacings = np.diff(evals)
    mean_spacing = np.mean(spacings)
    return spacings / (mean_spacing + 1e-10)  # Avoid division by zero

# Theoretical distributions
def wigner_dyson(s):
    """
    Wigner-Dyson distribution for delocalized states (GOE).
    Args:
        s: Normalized spacing
    Returns:
        P(s): Probability density
    """
    return (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)

def poisson(s):
    """
    Poisson distribution for localized states.
    Args:
        s: Normalized spacing
    Returns:
        P(s): Probability density
    """
    return np.exp(-s)

# Compute and plot level spacing statistics
plt.figure(figsize=(12, 4))
cmap = plt.cm.viridis  # Gradient colormap for histograms
s_range = np.linspace(0, 5, 200)  # Range for theoretical curves
for idx, W in enumerate(W_vals, 1):
    spacings = []
    # Collect spacings over realizations
    for r in range(num_realizations):
        sys.stdout.write(f'\rComputing W={W}, realization {r+1}/{num_realizations}')
        sys.stdout.flush()
        H = build_hamiltonian(N, W, t)
        evals = np.linalg.eigh(H)[0]  # Only need eigenvalues
        spacings.extend(level_statistics(evals))
    sys.stdout.write('\n')
    spacings = np.array(spacings)
    
    # Plot histogram with gradient color
    plt.subplot(1, 3, idx)
    hist, bins, _ = plt.hist(spacings, bins=num_bins, density=True, alpha=0.7,
                             color=cmap(idx/len(W_vals)), label='Numerical')
    plt.plot(s_range, poisson(s_range), 'k--', linewidth=2, label='Poisson')
    if W < 1:  # Wigner-Dyson only for weak disorder
        plt.plot(s_range, wigner_dyson(s_range), 'r--', linewidth=2, label='Wigner-Dyson')
    plt.xlabel(r'Normalized Spacing $s$')
    plt.ylabel(r'$P(s)$')
    plt.title(f'$W = {W}$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 5)
    plt.ylim(0, 1.5)
    plt.legend()

plt.suptitle(r'Level Spacing Distribution for Different Disorder Strengths', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()