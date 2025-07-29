import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import sys

# Configuration
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Paramètres
t = 1.0
N = 500
W_vals = [0, 0.5, 5.0, 10.0, 100.0, 10000.0]  # Sans W=2
num_realizations = 10

def build_hamiltonian(N, W, t):
    eps = np.random.uniform(-W/2, W/2, N)
    diagonals = [eps, -t*np.ones(N-1), -t*np.ones(N-1)]
    offsets = [0, 1, -1]
    return diags(diagonals, offsets, shape=(N, N)).toarray()

def calc_ipr(psi):
    psi_abs = np.abs(psi)
    psi_norm = np.sum(psi_abs**2, axis=0)
    mask = psi_norm > 1e-10
    psi_normalized = psi / np.sqrt(psi_norm + 1e-10)
    ipr = np.sum(psi_normalized**4, axis=0)
    ipr[~mask] = 0.0
    return ipr

# === Tracé des IPR ===
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()
cmap = plt.cm.viridis

for idx, W in enumerate(W_vals):
    iprs_all = np.zeros((num_realizations, N))
    evals_all = np.zeros((num_realizations, N))

    for r in range(num_realizations):
        sys.stdout.write(f'\rCalcul W={W}, réalisation {r+1}/{num_realizations}')
        sys.stdout.flush()
        H = build_hamiltonian(N, W, t)
        evals, evecs = np.linalg.eigh(H)
        iprs_all[r, :] = calc_ipr(evecs)
        evals_all[r, :] = evals
    sys.stdout.write('\n')

    iprs_avg = np.mean(iprs_all, axis=0)
    evals_avg = np.mean(evals_all, axis=0)

    ax = axs[idx]
    sc = ax.scatter(evals_avg, iprs_avg, s=5, alpha=0.6, c=iprs_avg, cmap=cmap)
    ax.set_xlabel(r'Énergie $E$')
    ax.set_ylabel(r'IPR')
    ax.set_title(f'$W = {W}$')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(iprs_avg)*1.1 if max(iprs_avg) > 0 else 1.0)
    plt.colorbar(sc, ax=ax, label='IPR')

fig.suptitle(r'Inverse Participation Ratio vs. Énergie', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# === Tracé des fonctions d'onde ===
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

for idx, W in enumerate(W_vals):
    H = build_hamiltonian(N, W, t)
    evals, evecs = np.linalg.eigh(H)
    idx_e = np.argmin(np.abs(evals))
    psi = np.abs(evecs[:, idx_e])**2

    ax = axs[idx]
    ax.plot(np.arange(N), psi, color=colors[idx], linewidth=1, alpha=0.8)
    ax.set_xlabel(r'Site $n$')
    ax.set_ylabel(r'$|\psi_n|^2$')
    ax.set_title(f'Fonction d\'onde ($W = {W}$, $E \\approx {evals[idx_e]:.2f}$)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_yscale('log')
    if W >= 5:
        ax.set_ylim(1e-30, 1e1)
    else:
        ax.set_ylim(1e-10, 1.0)

fig.suptitle(r'Fonctions d\'onde localisées pour différents $W$', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
