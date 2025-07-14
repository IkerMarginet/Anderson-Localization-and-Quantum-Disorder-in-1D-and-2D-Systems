import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns
import warnings
import logging
import os
from joblib import Parallel, delayed
import sys

# Check dependencies
try:
    import PIL
except ImportError:
    raise ImportError("Pillow is required for image processing. Install with `pip install pillow`.")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set Seaborn and Matplotlib styles
sns.set(style="whitegrid", palette="viridis")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
cmap = plt.cm.viridis

# Define the 2D Anderson Hamiltonian with sparse matrix
def anderson_hamiltonian_2d(N, W, t=1.0):
    """Construct the 2D Anderson model Hamiltonian with random disorder."""
    N_total = N * N
    disorder = np.random.uniform(-W/2, W/2, size=N_total)
    rows, cols, data = [], [], []
    for i in range(N):
        for j in range(N):
            k = i * N + j
            rows.append(k); cols.append(k); data.append(disorder[k])
            rows.append(k); cols.append(i * N + (j - 1) % N); data.append(-t)
            rows.append(k); cols.append(i * N + (j + 1) % N); data.append(-t)
            rows.append(k); cols.append(((i - 1) % N) * N + j); data.append(-t)
            rows.append(k); cols.append(((i + 1) % N) * N + j); data.append(-t)
    H = csr_matrix((data, (rows, cols)), shape=(N_total, N_total))
    return H, disorder.reshape(N, N)

# IPR calculation
def compute_ipr(eigenstates):
    """Calculate Inverse Participation Ratio (IPR)."""
    psi_abs = np.abs(eigenstates)
    psi_norm = np.sum(psi_abs**2, axis=0)
    mask = psi_norm > 1e-12
    psi_normalized = np.zeros_like(psi_abs)
    psi_normalized[:, mask] = psi_abs[:, mask] / np.sqrt(psi_norm[mask])
    ipr = np.sum(psi_normalized**4, axis=0)
    ipr[~mask] = 0.0
    return ipr

# Participation Ratio calculation
def compute_pr(eigenstates):
    """Calculate Participation Ratio (PR)."""
    ipr = compute_ipr(eigenstates)
    pr = np.where(ipr > 1e-12, 1 / ipr, 0.0)
    return pr

# Lyapunov Exponent calculation for 2D
def compute_lyapunov_2d(eigenstates, W, N):
    """Calculate Lyapunov exponent via 2D wavefunction radial decay."""
    if W == 0:
        return np.zeros(eigenstates.shape[1])
    le = []
    i, j = np.indices((N, N))
    for idx, psi in enumerate(eigenstates.T):
        psi_2d = psi.reshape(N, N)
        i_max, j_max = np.unravel_index(np.argmax(np.abs(psi_2d)**2), (N, N))
        di = np.minimum(np.abs(i - i_max), N - np.abs(i - i_max))
        dj = np.minimum(np.abs(j - j_max), N - np.abs(j - j_max))
        r = np.sqrt(di**2 + dj**2)
        log_psi = np.log(np.abs(psi_2d) + 1e-12)
        mask = r > 0
        try:
            slope, _ = np.polyfit(r[mask], log_psi[mask], 1)
            le.append(-slope if slope < 0 else 0.0)
        except Exception as e:
            logging.warning(f"Lyapunov fit failed for state {idx}: {e}")
            le.append(0.0)
    return np.array(le)

# Level spacing statistics
def level_spacing_statistics(energies, ax, label, color):
    """Plot level spacing distribution with theoretical comparisons."""
    sorted_E = np.sort(energies)
    s = np.diff(sorted_E)
    if len(s) > 1:
        s_mean = s / np.mean(s)
        bins = 50  # Fixed number of bins for better resolution
        sns.histplot(s_mean, bins=bins, stat='density', alpha=0.6, color=color, label=label, ax=ax)
        x = np.linspace(0, 3, 200)
        ax.plot(x, np.exp(-x), 'k--', label='Poisson (Localized)', linewidth=2)
        ax.plot(x, (np.pi/2)*x*np.exp(-np.pi*x**2/4), 'r--', label='Wigner-Dyson (Extended)', linewidth=2)
        ax.set_xlabel("Normalized Level Spacing, s")
        ax.set_ylabel("Probability Density, P(s)")
        ax.legend()

# Localization length fit for 2D
def fit_localization_length_2d(psi, N):
    """Fit localization length from 2D wavefunction decay."""
    psi_2d = psi.reshape(N, N)
    prob = np.abs(psi_2d)**2
    i_max, j_max = np.unravel_index(np.argmax(prob), (N, N))
    i, j = np.indices((N, N))
    di = np.minimum(np.abs(i - i_max), N - np.abs(i - i_max))
    dj = np.minimum(np.abs(j - j_max), N - np.abs(j - j_max))
    r = np.sqrt(di**2 + dj**2)
    log_prob = np.log(prob + 1e-12)
    mask = r > 0
    def exp_decay(r, A, xi):
        return A - 2 * r / xi
    try:
        popt, pcov = curve_fit(exp_decay, r[mask], log_prob[mask], p0=(0, N))
        xi = popt[1] if popt[1] > 0 else N
        residuals = log_prob[mask] - exp_decay(r[mask], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_prob[mask] - np.mean(log_prob[mask]))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-12)
        if r_squared < 0.5:
            logging.warning(f"Poor localization length fit (R²={r_squared:.2f})")
        fit_curve = exp_decay(r, *popt)
    except Exception as e:
        logging.warning(f"Localization length fit failed: {e}")
        xi = N
        fit_curve = np.zeros_like(r)
    return xi, (r, prob, fit_curve)

# Theoretical localization length (approximate for 2D)
def theoretical_localization_length(W, t=1.0):
    """Approximate theoretical localization length for 2D."""
    if W == 0:
        return float('inf')
    return np.exp(2 * t / (W + 1e-12))

# Time evolution
def time_evolution(H, psi0, times):
    """Compute time evolution of wavefunction using sparse matrix."""
    psi_t = []
    for t_val in times:
        U_psi = expm_multiply(-1j * H * t_val, psi0)
        psi_t.append(np.abs(U_psi)**2)
    return np.array(psi_t)

# Density of States calculation
def compute_dos(energies, sigma=0.1):
    """Compute density of states using Gaussian kernel density estimation."""
    kde = gaussian_kde(energies, bw_method=sigma)
    x = np.linspace(np.min(energies), np.max(energies), 500)
    y = kde(x)
    return x, y

# Theoretical DOS for clean 2D system (simplified approximation)
def theoretical_dos_2d(E, t=1.0):
    """Simplified theoretical DOS for clean 2D tight-binding model."""
    E = np.clip(E, -4*t, 4*t)
    return 1 / (np.pi**2 * t) * np.ones_like(E) / np.sqrt(1 - (E/(4*t))**2 + 1e-12)

# Plot DOS comparison
def plot_dos_comparison(evals, W, t=1.0, output_dir='output'):
    """Create enhanced DOS comparison plot."""
    fig = plt.figure(figsize=(12, 8))
    hist, bin_edges = np.histogram(evals, bins=200, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_centers, hist, width=bin_width * 0.9, color='blue', alpha=0.7, label='Numerical Histogram')
    x_kde, y_kde = compute_dos(evals.flatten())
    plt.plot(x_kde, y_kde, 'red', linewidth=2, label='Numerical KDE')
    if W == 0:
        x_theory = np.linspace(-4*t, 4*t, 200)
        y_theory = theoretical_dos_2d(x_theory, t)
        plt.plot(x_theory, y_theory, 'green', linewidth=2, label='Theoretical DOS (Approx)')
    plt.xlabel('Energy E')
    plt.ylabel('Density of States')
    plt.title(f'DOS Comparison (W={W})')
    plt.legend()
    save_plot(fig, f'dos_comparison_W{W}.png', output_dir)

# 3D Plot for Disorder Profile
def plot_3d_disorder(disorder, W, N, output_dir='output'):
    """Create a smooth 3D surface plot of the disorder profile."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    # Interpolate to a finer grid
    fine_N = 100
    fine_x = np.linspace(0, N-1, fine_N)
    fine_y = np.linspace(0, N-1, fine_N)
    fine_X, fine_Y = np.meshgrid(fine_x, fine_y)
    interpolator = RegularGridInterpolator((np.arange(N), np.arange(N)), disorder, method='cubic')
    Z = interpolator((fine_X, fine_Y))
    ax.plot_surface(fine_X, fine_Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V(x,y)')
    ax.set_title(f'Disorder Profile (W={W})')
    save_plot(fig, f'disorder_W{W}.png', output_dir)

# 3D Plot for Wavefunction
def plot_3d_wavefunction(state, energy, W, N, output_dir='output'):
    """Create a smooth 3D surface plot of the wavefunction probability density."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    Z = np.abs(state.reshape(N, N))**2
    # Interpolate to a finer grid
    fine_N = 100
    fine_x = np.linspace(0, N-1, fine_N)
    fine_y = np.linspace(0, N-1, fine_N)
    fine_X, fine_Y = np.meshgrid(fine_x, fine_y)
    interpolator = RegularGridInterpolator((np.arange(N), np.arange(N)), Z, method='cubic')
    fine_Z = interpolator((fine_X, fine_Y))
    ax.plot_surface(fine_X, fine_Y, fine_Z, cmap=cmap, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('|ψ(x,y)|²')
    ax.set_title(f'Wavefunction |ψ|² (W={W}, E={energy:.2f})')
    save_plot(fig, f'wavefunction_W{W}.png', output_dir)

# 3D Animation for Time Evolution
def create_time_evolution_animation_3d(H, N, W, num_frames=100, duration=10, output_dir='output'):
    """Create and save a smooth 3D animation of the wavefunction time evolution."""
    psi0 = np.zeros(N * N, dtype=complex)
    center_idx = (N // 2) * N + N // 2
    psi0[center_idx] = 1.0
    times = np.linspace(0, duration, num_frames)
    prob_t = time_evolution(H, psi0, times)
    max_prob = np.max(prob_t)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    # Prepare interpolation grid
    fine_N = 100
    fine_x = np.linspace(0, N-1, fine_N)
    fine_y = np.linspace(0, N-1, fine_N)
    fine_X, fine_Y = np.meshgrid(fine_x, fine_y)
    
    Z = prob_t[0].reshape(N, N)
    interpolator = RegularGridInterpolator((np.arange(N), np.arange(N)), Z, method='cubic')
    fine_Z = interpolator((fine_X, fine_Y))
    surf = ax.plot_surface(fine_X, fine_Y, fine_Z, cmap='viridis', edgecolor='none')
    ax.set_zlim(0, max_prob * 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('|ψ|²')
    ax.set_title(f'Time Evolution (W={W}, t=0.00)')
    
    def animate(i):
        ax.clear()
        Z = prob_t[i].reshape(N, N)
        interpolator = RegularGridInterpolator((np.arange(N), np.arange(N)), Z, method='cubic')
        fine_Z = interpolator((fine_X, fine_Y))
        surf = ax.plot_surface(fine_X, fine_Y, fine_Z, cmap='viridis', edgecolor='none')
        ax.set_zlim(0, max_prob * 1.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('|ψ|²')
        ax.set_title(f'Time Evolution (W={W}, t={times[i]:.2f})')
        return surf,
    
    ani = FuncAnimation(fig, animate, frames=num_frames, blit=False, interval=50)
    writer = PillowWriter(fps=20)
    filepath = os.path.join(output_dir, f'time_evolution_3d_W{W}.gif')
    if os.path.exists(filepath):
        logging.warning(f"Overwriting {filepath}")
    ani.save(filepath, writer=writer)
    plt.close()

# Helper function to save plots
def save_plot(fig, filename, output_dir):
    """Save a plot to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        logging.warning(f"Overwriting {filepath}")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)

# Single realization computation
def run_realization(N, W, t, r):
    """Compute metrics for a single realization."""
    H, disorder = anderson_hamiltonian_2d(N, W, t)
    evals, evecs = eigsh(H, k=N*N-2, which='LM')  # Use sparse solver
    iprs = compute_ipr(evecs)
    le = compute_lyapunov_2d(evecs, W, N)
    pr = compute_pr(evecs)
    return iprs, le, pr, evals, evecs, H, disorder

# Main simulation
def run_simulation(N=20, t=1.0, W_vals=[0, 0.5, 1, 2, 5, 10], num_realizations=5, output_dir='output'):
    """Run the 2D Anderson localization simulation."""
    os.makedirs(output_dir, exist_ok=True)
    colors = sns.color_palette("viridis", len(W_vals))
    
    # Store data for all W values
    all_iprs_mean = []
    all_iprs_std = []
    all_le_mean = []
    all_le_std = []
    all_evals_avg = []
    all_evecs_last = []
    all_H_last = []
    all_disorder_last = []
    
    for W_idx, W in enumerate(W_vals):
        print(f"\n--- Simulating W = {W} ---")
        results = Parallel(n_jobs=-1)(delayed(run_realization)(N, W, t, r) for r in range(num_realizations))
        iprs_all = np.array([res[0] for res in results])
        le_all = np.array([res[1] for res in results])
        pr_all = np.array([res[2] for res in results])
        evals_all = np.array([res[3] for res in results])
        evecs_last, H_last, disorder_last = results[-1][4], results[-1][5], results[-1][6]
        
        print(f"\rCompleted {num_realizations} realizations")
        
        # Compute averages and standard errors
        evals_avg = np.mean(evals_all, axis=0)
        iprs_mean = np.mean(iprs_all, axis=0)
        iprs_std = np.std(iprs_all, axis=0) / np.sqrt(num_realizations)
        le_mean = np.mean(le_all, axis=0)
        le_std = np.std(le_all, axis=0) / np.sqrt(num_realizations)
        
        # Sort data by energy
        sorted_indices = np.argsort(evals_avg)
        evals_sorted = evals_avg[sorted_indices]
        iprs_mean_sorted = iprs_mean[sorted_indices]
        iprs_std_sorted = iprs_std[sorted_indices]
        le_mean_sorted = le_mean[sorted_indices]
        le_std_sorted = le_std[sorted_indices]
        
        all_evals_avg.append(evals_sorted)
        all_iprs_mean.append(iprs_mean_sorted)
        all_iprs_std.append(iprs_std_sorted)
        all_le_mean.append(le_mean_sorted)
        all_le_std.append(le_std_sorted)
        all_evecs_last.append(evecs_last)
        all_H_last.append(H_last)
        all_disorder_last.append(disorder_last)
        
        # Individual IPR Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(evals_sorted, iprs_mean_sorted, s=50, color=colors[W_idx], alpha=0.6, label=f'W={W}')
        ax.set_xlabel('Energy E')
        ax.set_ylabel('IPR')
        ax.set_yscale('log')
        ax.set_title(f'IPR vs Energy (W={W})')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(loc='upper right')
        save_plot(fig, f'ipr_W{W}.png', output_dir)
        
        # Individual Lyapunov Exponent Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(evals_sorted, le_mean_sorted, s=50, marker='s', color=colors[W_idx], alpha=0.6, label=f'W={W}')
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Lyapunov Exponent γ')
        ax.set_yscale('linear')
        ax.set_title(f'Lyapunov Exponent vs Energy (W={W})')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(loc='upper right')
        save_plot(fig, f'lyapunov_W{W}.png', output_dir)
        
        # Disorder profile (3D plot)
        plot_3d_disorder(disorder_last, W, N, output_dir)
        
        # Level spacing
        fig, ax = plt.subplots(figsize=(10, 6))
        level_spacing_statistics(evals_avg, ax, f'W={W}', colors[W_idx])
        plt.title(f'Level Spacing (W={W})')
        save_plot(fig, f'level_spacing_W{W}.png', output_dir)
        
        # DOS
        plot_dos_comparison(evals_all, W, t, output_dir)
        
        # Localization length
        idx_most_localized = np.argmax(iprs_mean)
        xi_fit, (r, prob, fit_curve) = fit_localization_length_2d(evecs_last[:, idx_most_localized], N)
        xi_theory = theoretical_localization_length(W, t)
        
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(r.flatten(), prob.flatten(), s=50, color=colors[W_idx], alpha=0.6, label='|ψ(x,y)|²')
        if np.any(fit_curve):
            plt.plot(r.flatten(), np.exp(fit_curve.flatten()), 'k-', lw=2, label=f'Fit: ξ={xi_fit:.2f}')
        plt.yscale('log')
        plt.xlabel('Radial Distance r')
        plt.ylabel('|ψ(x,y)|²')
        plt.title(f'Localization Fit (W={W}, ξ_theory={xi_theory:.2f})')
        plt.legend()
        save_plot(fig, f'loc_length_fit_W{W}.png', output_dir)
        
        # Wavefunction (3D plot)
        plot_3d_wavefunction(evecs_last[:, idx_most_localized], evals_avg[idx_most_localized], W, N, output_dir)
        
        # Time evolution (3D animation)
        create_time_evolution_animation_3d(H_last, N, W, output_dir=output_dir)
        
        print(f"  Fitted ξ = {xi_fit:.2f}, Theoretical ξ = {xi_theory if W != 0 else 'inf'}")
    
    # Combined IPR Plot for All W Values
    fig, ax = plt.subplots(figsize=(14, 8))
    for W_idx, W in enumerate(W_vals):
        evals_sorted = all_evals_avg[W_idx]
        iprs_mean_sorted = all_iprs_mean[W_idx]
        iprs_std_sorted = all_iprs_std[W_idx]
        
        # Plot IPR with error bars
        ax.errorbar(evals_sorted, iprs_mean_sorted, yerr=iprs_std_sorted, fmt='o', 
                    color=colors[W_idx], alpha=0.6, label=f'W={W}', capsize=3)
    
    # Customize axes
    ax.set_xlabel('Energy E')
    ax.set_ylabel('IPR')
    ax.set_yscale('log')
    ax.set_title('IPR vs Energy for All W')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save the IPR plot
    save_plot(fig, 'ipr_combined_all_W.png', output_dir)
    
    # Combined Lyapunov Exponent Plot for All W Values
    fig, ax = plt.subplots(figsize=(14, 8))
    for W_idx, W in enumerate(W_vals):
        evals_sorted = all_evals_avg[W_idx]
        le_mean_sorted = all_le_mean[W_idx]
        le_std_sorted = all_le_std[W_idx]
        
        # Plot Lyapunov Exponent with error bars
        ax.errorbar(evals_sorted, le_mean_sorted, yerr=le_std_sorted, fmt='s', 
                    color=colors[W_idx], alpha=0.6, label=f'W={W}', capsize=3)
    
    # Customize axes
    ax.set_xlabel('Energy E')
    ax.set_ylabel('Lyapunov Exponent γ')
    ax.set_yscale('linear')
    ax.set_title('Lyapunov Exponent vs Energy for All W')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save the Lyapunov plot
    save_plot(fig, 'lyapunov_combined_all_W.png', output_dir)

if __name__ == "__main__":
    run_simulation()