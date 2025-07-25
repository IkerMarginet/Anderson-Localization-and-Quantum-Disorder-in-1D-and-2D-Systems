import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn, jn_zeros
from scipy.linalg import eigh
import os

# --- Style ---
sns.set(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3
})

# --- Create folder structure like stark.py ---
def create_directories(base="bessel_anderson_results"):
    subdirs = ["Bessel", "Optics", "Wavefunction"]
    os.makedirs(base, exist_ok=True)
    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base

# --- 1. Plot Bessel functions ---
def plot_bessel_functions(base_dir):
    x = np.linspace(0, 20, 1000)
    plt.figure()
    for n in range(4):
        plt.plot(x, jn(n, x), label=f"$J_{n}(x)$", linewidth=2)
    plt.title("Bessel Functions of the First Kind")
    plt.xlabel("x")
    plt.ylabel("$J_n(x)$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Bessel", "bessel_functions.png"), dpi=300)
    plt.close()

# --- 2. Optical Fiber Cutoff Modes ---
def plot_bessel_cutoffs(base_dir):
    x = np.linspace(0, 20, 1000)
    zeros = jn_zeros(0, 5)
    plt.figure()
    plt.plot(x, jn(0, x), label="$J_0(x)$", linewidth=2)
    for i, z in enumerate(zeros):
        plt.axvline(z, color='red', linestyle='--', alpha=0.6,
                    label=f"$x_{{0,{i+1}}} ≈ {z:.2f}$" if i == 0 else None)
    plt.axhline(0, color='gray', lw=1, linestyle='--')
    plt.title("Optical Fiber Cutoffs: Zeros of $J_0(x)$")
    plt.xlabel("x")
    plt.ylabel("$J_0(x)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Optics", "optical_cutoffs.png"), dpi=300)
    plt.close()

# --- 3. Anderson Localization ---
def simulate_anderson_localization(N=500, W=2.5, seed=42):
    np.random.seed(seed)
    H = np.zeros((N, N))
    diag = W * (2 * np.random.rand(N) - 1)
    for i in range(N):
        H[i, i] = diag[i]
        if i > 0:
            H[i, i - 1] = H[i - 1, i] = -1
    eigvals, eigvecs = eigh(H)
    return eigvals, eigvecs

def plot_localized_states(eigvecs, base_dir, indices=[100, 150, 200]):
    plt.figure()
    for idx in indices:
        psi = np.abs(eigvecs[:, idx])**2
        plt.plot(psi, label=f"$|ψ_{{{idx}}}|^2$")
    plt.title("Anderson Localization: Selected Eigenstates")
    plt.xlabel("Site index")
    plt.ylabel("Probability density $|ψ|^2$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Wavefunction", "anderson_localization.png"), dpi=300)
    plt.close()

# --- Main execution ---
if __name__ == "__main__":
    base = create_directories()
    plot_bessel_functions(base)
    plot_bessel_cutoffs(base)
    _, eigvecs = simulate_anderson_localization()
    plot_localized_states(eigvecs, base)
