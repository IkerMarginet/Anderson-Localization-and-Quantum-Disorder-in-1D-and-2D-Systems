import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy import linalg
from joblib import Parallel, delayed
import warnings

@jit(nopython=True)
def _Σ_discrete_iter(E, σ2, t, k_vals, tol, max_iter):
    Σ = 1j * σ2 / 2
    for _ in range(max_iter):
        denom = E + 2 * t * np.cos(k_vals) - Σ
        new_Σ = σ2 * np.sum(1 / denom) / len(k_vals)
        if np.abs(new_Σ - Σ) < tol:
            return new_Σ
        Σ = new_Σ
    return Σ

@jit(nopython=True)
def _Σ_integral_iter(E, σ2, t, tol, max_iter):
    Σ = 1j * σ2 / 2
    for _ in range(max_iter):
        z_disc = (E - Σ)**2 - 4 * t**2
        sqrt_term = np.sqrt(z_disc)
        z1 = (E - Σ - sqrt_term) / (2 * t)
        z2 = (E - Σ + sqrt_term) / (2 * t)
        residue = 1 / (z1 - z2) if abs(z1) < 1 else 1 / (z2 - z1)
        new_Σ = σ2 * residue / t
        if np.abs(new_Σ - Σ) < tol:
            return new_Σ
        Σ = new_Σ
    return Σ

@jit(nopython=True)
def _Σ_secant_iter(E, σ2, t, k_vals, tol, max_iter):
    Σ0, Σ1 = 1j * σ2 / 2, 1j * σ2 / 2 + 1e-6
    for _ in range(max_iter):
        f0 = Σ0 - σ2 * np.sum(1 / (E + 2 * t * np.cos(k_vals) - Σ0)) / len(k_vals)
        f1 = Σ1 - σ2 * np.sum(1 / (E + 2 * t * np.cos(k_vals) - Σ1)) / len(k_vals)
        if abs(f1) < tol:
            return Σ1
        Σ2 = Σ1 - f1 * (Σ1 - Σ0) / (f1 - f0 + 1e-10)  # Add small value to avoid division by zero
        if abs(Σ2 - Σ1) < tol:
            return Σ2
        Σ0, Σ1 = Σ1, Σ2
    return Σ1

class Anderson1D:
    def __init__(self, W=1.0, t=1.0, N=1000, η=1e-8, tol=1e-10, max_iter=500):
        self.W = W
        self.t = t
        self.N = N
        self.η = η
        self.tol = tol
        self.max_iter = max_iter
        self.σ2 = W**2 / 12
        self.k_vals = 2 * np.pi * np.arange(N) / N

    def Σ_discrete(self, E):
        try:
            return _Σ_discrete_iter(E + 1j * self.η, self.σ2, self.t, self.k_vals, self.tol, self.max_iter)
        except Exception as e:
            warnings.warn(f"Discrete method failed for E={E}: {str(e)}")
            return np.nan + 1j * np.nan

    def Σ_integral(self, E):
        try:
            return _Σ_integral_iter(E + 1j * self.η, self.σ2, self.t, self.tol, self.max_iter)
        except Exception as e:
            warnings.warn(f"Integral method failed for E={E}: {str(e)}")
            return np.nan + 1j * np.nan

    def Σ_secant(self, E):
        try:
            return _Σ_secant_iter(E + 1j * self.η, self.σ2, self.t, self.k_vals, self.tol, self.max_iter)
        except Exception as e:
            warnings.warn(f"Secant method failed for E={E}: {str(e)}")
            return np.nan + 1j * np.nan

    def diagonalize_random_H(self, N_small=100):
        """Exact diagonalization for small systems"""
        try:
            H = np.diag(np.random.uniform(-self.W/2, self.W/2, N_small)) + \
                self.t * (np.diag(np.ones(N_small-1), 1) + np.diag(np.ones(N_small-1), -1))
            eigenvalues = linalg.eigvalsh(H)
            return eigenvalues
        except Exception as e:
            warnings.warn(f"Diagonalization failed: {str(e)}")
            return np.array([])

    def compute_dos(self, E_vals, method='integral'):
        def compute_single_E(E):
            try:
                if method == 'integral':
                    Σ = self.Σ_integral(E)
                elif method == 'discrete':
                    Σ = self.Σ_discrete(E)
                else:
                    Σ = self.Σ_secant(E)
                ρ = max(np.imag(Σ), 0) / (np.pi * self.σ2)
                return Σ, ρ
            except:
                return np.nan + 1j * np.nan, np.nan

        results = Parallel(n_jobs=-1)(delayed(compute_single_E)(E) for E in E_vals)
        Σ_list, ρ_list = zip(*results)
        return np.array(Σ_list), np.array(ρ_list)

    def analyze_methods(self, E_vals):
        Σ_disc, ρ_disc = self.compute_dos(E_vals, method='discrete')
        Σ_int, ρ_int = self.compute_dos(E_vals, method='integral')
        Σ_sec, ρ_sec = self.compute_dos(E_vals, method='secant')
        return Σ_disc, Σ_int, Σ_sec, ρ_disc, ρ_int, ρ_sec

    def compute_localization_length(self, E_vals):
        """Compute localization length from Im[Σ]"""
        _, ρ_int = self.compute_dos(E_vals, method='integral')
        loc_length = 1 / np.where(ρ_int > 0, np.log(ρ_int + 1e-10), np.inf)  # Add small value to avoid log(0)
        return loc_length

    def plot_comparison(self, E_vals, Σ_d, Σ_i, Σ_s, ρ_d, ρ_i, ρ_s):
        fig, axs = plt.subplots(4, 1, figsize=(10, 16))
        
        # Real part of Σ
        axs[0].plot(E_vals, Σ_d.real, label="Discrete Re[Σ]", color='b')
        axs[0].plot(E_vals, Σ_i.real, label="Integral Re[Σ]", color='r', linestyle='--')
        axs[0].plot(E_vals, Σ_s.real, label="Secant Re[Σ]", color='g', linestyle=':')
        axs[0].set_ylabel("Re[Σ(E)]")
        axs[0].legend()
        axs[0].grid(True)

        # Imaginary part of Σ
        axs[1].plot(E_vals, Σ_d.imag, label="Discrete Im[Σ]", color='b')
        axs[1].plot(E_vals, Σ_i.imag, label="Integral Im[Σ]", color='r', linestyle='--')
        axs[1].plot(E_vals, Σ_s.imag, label="Secant Im[Σ]", color='g', linestyle=':')
        axs[1].set_ylabel("Im[Σ(E)]")
        axs[1].legend()
        axs[1].grid(True)

        # Density of States
        axs[2].plot(E_vals, ρ_d, label="Discrete ρ(E)", color='b')
        axs[2].plot(E_vals, ρ_i, label="Integral ρ(E)", color='r', linestyle='--')
        axs[2].plot(E_vals, ρ_s, label="Secant ρ(E)", color='g', linestyle=':')
        eigenvalues = self.diagonalize_random_H()
        if len(eigenvalues) > 0:
            axs[2].hist(eigenvalues, bins=50, density=True, alpha=0.3, label="Exact diag.", color='k')
        axs[2].set_xlabel("E")
        axs[2].set_ylabel("Density of States ρ(E)")
        axs[2].legend()
        axs[2].grid(True)

        # Localization length
        loc_length = self.compute_localization_length(E_vals)
        axs[3].plot(E_vals, loc_length, label="Localization Length", color='m')
        axs[3].set_xlabel("E")
        axs[3].set_ylabel("Localization Length")
        axs[3].legend()
        axs[3].grid(True)

        plt.tight_layout()
        plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')  # Save figure with a filename
        plt.show()

    def plot_disorder_variation(self, E_vals, W_vals, dist_type='uniform'):
        plt.figure(figsize=(10, 6))
        for W in W_vals:
            self.W = W
            if dist_type == 'uniform':
                self.σ2 = W**2 / 12
            elif dist_type == 'gaussian':
                self.σ2 = W**2
            else:
                raise ValueError("Unsupported distribution type")
            Σ, ρ = self.compute_dos(E_vals, method='integral')
            plt.plot(E_vals, ρ, label=f"W={W}, {dist_type}")
        plt.xlabel("E")
        plt.ylabel("ρ(E)")
        plt.title("Density of States for Various Disorder Strengths")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'disorder_variation_{dist_type}.png', dpi=300, bbox_inches='tight')  # Save with distribution-specific filename
        plt.show()

# Run analysis
W, t, N = 1.0, 1.0, 1000
model = Anderson1D(W=W, t=t, N=N)
E_vals = np.linspace(-2.5, 2.5, 200)
Σ_d, Σ_i, Σ_s, ρ_d, ρ_i, ρ_s = model.analyze_methods(E_vals)
model.plot_comparison(E_vals, Σ_d, Σ_i, Σ_s, ρ_d, ρ_i, ρ_s)
model.plot_disorder_variation(E_vals, W_vals=[0.5, 1.0, 2.0, 4.0], dist_type='uniform')
model.plot_disorder_variation(E_vals, W_vals=[0.5, 1.0, 2.0, 4.0], dist_type='gaussian')