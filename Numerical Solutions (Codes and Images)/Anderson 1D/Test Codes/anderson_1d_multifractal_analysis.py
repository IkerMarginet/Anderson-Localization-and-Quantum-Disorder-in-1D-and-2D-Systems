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
num_realizations = 10  # Number of disorder realizations
q_vals = np.linspace(0.1, 5.0, 20)  # Moment orders for multifractal analysis

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
    eps = np.random.uniform(-W/2, W/2, N)
    diagonals = [eps, -t*np.ones(N-1), -t*np.ones(N-1)]
    offsets = [0, 1, -1]
    return diags(diagonals, offsets, shape=(N, N)).toarray()

def calc_multifractal_exponent(psi, q_vals):
    """
    Calculate the multifractal exponent tau(q) for a wavefunction.
    tau(q) = -ln(sum(|psi_n|^(2q))) / ln(N)
    Args:
        psi: Wavefunction (N x 1)
        q_vals: Array of moment orders
    Returns:
        tau: Multifractal exponent tau(q)
    """
    psi_abs = np.abs(psi)
    psi_norm = np.sum(psi_abs**2)
    if psi_norm < 1e-10:
        return np.zeros_like(q_vals)
    psi_normalized = psi / np.sqrt(psi_norm)
    moments = np.array([np.sum(np.abs(psi_normalized)**(2*q)) for q in q_vals])
    # Avoid log of zero or negative
    moments = np.clip(moments, 1e-10, None)
    tau = -np.log(moments) / np.log(N)
    return tau

# Compute and plot multifractal exponents
plt.figure(figsize=(12, 4))
cmap = plt.cm.viridis  # Gradient colormap
colors = ['blue', 'red', 'green']  # Colors for different W
for idx, W in enumerate(W_vals, 1):
    tau_all = np.zeros((num_realizations, len(q_vals)))
    # Collect tau(q) for eigenstate near E=0
    for r in range(num_realizations):
        sys.stdout.write(f'\rComputing W={W}, realization {r+1}/{num_realizations}')
        sys.stdout.flush()
        H = build_hamiltonian(N, W, t)
        evals, evecs = np.linalg.eigh(H)
        # Select eigenstate closest to E=0
        idx_e = np.argmin(np.abs(evals))
        psi = evecs[:, idx_e]
        tau_all[r, :] = calc_multifractal_exponent(psi, q_vals)
    sys.stdout.write('\n')
    
    # Average over realizations
    tau_avg = np.mean(tau_all, axis=0)
    
    # Plot tau(q) with distinct colors
    plt.subplot(1, 3, idx)
    plt.plot(q_vals, tau_avg, color=colors[idx-1], linewidth=2, alpha=0.8,
             label=f'$W={W}$')
    # Reference line for delocalized states: tau(q) = q-1
    plt.plot(q_vals, q_vals - 1, 'k--', label=r'$\tau(q) = q-1$ (Delocalized)')
    plt.xlabel(r'Moment Order $q$')
    plt.ylabel(r'Multifractal Exponent $\tau(q)$')
    plt.title(f'Multifractal Analysis ($W = {W}$)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 5)
    plt.ylim(-1, 4)
    plt.legend()

plt.suptitle(r'Multifractal Exponent $\tau(q)$ for Different Disorder Strengths', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()