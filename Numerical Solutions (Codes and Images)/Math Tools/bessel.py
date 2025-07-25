import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros
from pathlib import Path

# Create output folder
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ---------------------------
# 1. Bessel Functions J_n(x)
# ---------------------------
x = np.linspace(0, 20, 1000)
plt.figure(figsize=(10, 6))
for n in range(4):
    plt.plot(x, jn(n, x), label=f"$J_{{{n}}}(x)$", linewidth=2)
plt.title("Bessel Functions of the First Kind", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("$J_n(x)$", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "bessel_functions.png", dpi=300)
plt.close()

# ---------------------------------------
# 2. Optical Application: Mode Cutoff
# ---------------------------------------
zeros_j0 = jn_zeros(0, 5)
plt.figure(figsize=(10, 6))
plt.plot(x, jn(0, x), label="$J_0(x)$", linewidth=2)
for i, zero in enumerate(zeros_j0):
    plt.axvline(zero, color='red', linestyle='--', alpha=0.6,
                label=f"$x_{{0,{i+1}}} ≈ {zero:.2f}$" if i < 1 else None)
plt.axhline(0, color='gray', lw=1, linestyle='--')
plt.title("Zeros of $J_0(x)$: Mode Cutoffs in Optical Fibers", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("$J_0(x)$", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "optical_fiber_modes.png", dpi=300)
plt.close()

# ---------------------------------------
# 3. Anderson Localization Simulation
# ---------------------------------------
def simulate_anderson(N=500, W=2.5, seed=42):
    np.random.seed(seed)
    H = np.zeros((N, N))
    on_site = W * (2 * np.random.rand(N) - 1)
    for i in range(N):
        H[i, i] = on_site[i]
        if i > 0:
            H[i, i - 1] = H[i - 1, i] = -1
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals, eigvecs

eigvals, eigvecs = simulate_anderson(N=500, W=2.5)

# Plot selected localized states
selected_indices = [100, 150, 200]
plt.figure(figsize=(10, 6))
for idx in selected_indices:
    psi = np.abs(eigvecs[:, idx])**2
    plt.plot(psi, label=f"$|ψ_{{{idx}}}|^2$", linewidth=1.8)
plt.title("Anderson Localization: Localized Eigenstates", fontsize=16)
plt.xlabel("Site Index", fontsize=14)
plt.ylabel("Probability Density $|ψ|^2$", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "anderson_localization.png", dpi=300)
plt.close()
