import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys

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

# Define the Maryland Hamiltonian
def maryland_hamiltonian(N, λ, φ=20.0, t=1.0, α=(np.sqrt(5)-1)/2):
    """
    Construct the 1D Maryland model Hamiltonian with tangent potential.
    
    Parameters:
    N - number of sites
    λ - disorder strength
    φ - phase of potential
    t - hopping amplitude
    α - irrational number (default golden ratio)
    """
    # Quasiperiodic potential with singularity handling
    n = np.arange(N)
    angles = 2 * np.pi * α * n + φ
    with np.errstate(invalid='ignore'):
        potential = λ * np.tan(angles)
    # Cap extremely large values and handle singularities
    potential = np.clip(potential, -1e10, 1e10)
    potential = np.nan_to_num(potential, nan=0.0)
    
    diagonal = potential
    off_diagonal = -t * np.ones(N-1)
    H = diags([off_diagonal, diagonal, off_diagonal], offsets=[-1, 0, 1]).toarray()
    return H, potential

# IPR calculation (unchanged)
def compute_ipr(eigenstates):
    """Calculate Inverse Participation Ratio (IPR)."""
    psi_abs = np.abs(eigenstates)
    psi_norm = np.sum(psi_abs**2, axis=0)
    mask = psi_norm > 1e-10
    psi_normalized = psi_abs / np.sqrt(psi_norm + 1e-10)
    ipr = np.sum(psi_normalized**4, axis=0)
    ipr[~mask] = 0.0
    return ipr

# Participation Ratio calculation (unchanged)
def compute_pr(eigenstates):
    """Calculate Participation Ratio (PR)."""
    ipr = compute_ipr(eigenstates)
    pr = 1 / (ipr + 1e-10)
    return pr

# Lyapunov Exponent calculation (unchanged)
def compute_lyapunov(eigenstates, λ):
    """Calculate Lyapunov exponent via wavefunction decay."""
    if λ == 0:
        return np.zeros(eigenstates.shape[1])  # All zeros for clean system
    
    le = []
    for psi in eigenstates.T:
        psi = psi / np.linalg.norm(psi)
        log_psi = np.log(np.abs(psi) + 1e-10)
        x = np.arange(len(psi))
        slope, _ = np.polyfit(x, log_psi, 1)
        le.append(-slope)
    return np.array(le)

# Level spacing statistics (unchanged)
def level_spacing_statistics(energies, ax, label, color):
    """Plot level spacing distribution with theoretical comparisons."""
    sorted_E = np.sort(energies)
    s = np.diff(sorted_E)
    if len(s) > 1:
        s_mean = s / np.mean(s)
        sns.histplot(s_mean, bins=500, stat='density', alpha=0.6, color=color, label=label, ax=ax)
        x = np.linspace(0, 3, 200)
        ax.plot(x, np.exp(-x), 'k--', label='Poisson (Localized)', linewidth=2)
        ax.plot(x, (np.pi/2)*x*np.exp(-np.pi*x**2/4), 'r--', label='Wigner-Dyson (Extended)', linewidth=2)
        ax.set_xlabel("Normalized Level Spacing, s")
        ax.set_ylabel("Probability Density, P(s)")
        ax.legend()

# Localization length fit (unchanged)
def fit_localization_length(psi):
    """Fit localization length from wavefunction decay."""
    x = np.arange(len(psi))
    prob = np.abs(psi)**2
    max_idx = np.argmax(prob)
    tail = prob[max_idx:]
    x_tail = x[max_idx:]
    def exp_decay(x, A, x0, xi):
        return A * np.exp(-np.abs(x - x0) / xi)
    try:
        popt, _ = curve_fit(exp_decay, x_tail, tail, p0=(1, max_idx, 10), maxfev=2000)
        return popt[2], (x, prob, exp_decay(x, *popt))
    except:
        return len(psi), (x, prob, np.zeros_like(x))

# Theoretical localization length for Maryland model
def theoretical_localization_length(λ, t=1.0):
    """Calculate theoretical localization length for Maryland model."""
    if λ == 0:
        return float('inf')
    else:
        # All states localized for any λ ≠ 0
        return 1 / np.log(1 + np.abs(λ)/2)  # Approximate formula

# Time evolution (unchanged)
def time_evolution(H, psi0, times):
    """Compute time evolution of wavefunction."""
    from scipy.linalg import expm
    psi_t = []
    for t_val in times:
        U = expm(-1j * H * t_val)
        psi_t.append(np.abs(U @ psi0)**2)
    return np.array(psi_t)

# Density of States calculation (unchanged)
def compute_dos(energies, sigma=0.1):
    """Compute density of states using Gaussian kernel density estimation."""
    kde = gaussian_kde(energies, bw_method=sigma)
    x = np.linspace(np.min(energies), np.max(energies), 500)
    y = kde(x)
    return x, y

# Theoretical DOS for clean system (unchanged)
def theoretical_dos(E, t=1.0):
    """Theoretical density of states for clean 1D system."""
    return 1 / (np.pi * np.sqrt(4 * t**2 - E**2 + 1e-10))

def plot_dos_comparison(evals, λ, t=1.0):
    """Create enhanced DOS comparison plot with high-resolution histogram."""
    plt.figure(figsize=(12, 8))
    
    # Energy range for plotting
    e_min, e_max = np.min(evals), np.max(evals)
    energy_range = e_max - e_min
    
    # Theoretical DOS for clean system
    x_theory = np.linspace(-2*t, 2*t, 200)
    y_theory = theoretical_dos(x_theory, t)
    
    # Use many more bins for better resolution (200 bins)
    n_bins = 500
    hist, bin_edges = np.histogram(evals, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Plot histogram with thin bars for high resolution
    plt.bar(bin_centers, hist, width=bin_width*0.9, 
            color='blue', alpha=0.7, edgecolor='black', linewidth=0.5,
            label=f'Numerical Histogram ({n_bins} bins)')
    
    # Plot KDE estimate
    x_kde, y_kde = compute_dos(evals.flatten())
    plt.plot(x_kde, y_kde, 'red', linewidth=2, 
             alpha=0.9, label='Numerical KDE')
    
    # Plot theoretical DOS for clean system
    if λ == 0:
        plt.plot(x_theory, y_theory, 'green', linewidth=3, 
                 alpha=0.8, label='Theoretical DOS')
    
    plt.xlabel('Energy E')
    plt.ylabel('Density of States')
    plt.title(f'High-Resolution DOS Comparison (λ={λ})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f"""
    Statistics (λ={λ}):
    Mean energy: {np.mean(evals):.2f}
    Energy std: {np.std(evals):.2f}
    Histogram integral: {np.sum(hist)*bin_width:.2f}
    KDE integral: {np.trapezoid(y_kde, x_kde):.2f}
    """
    if λ == 0:
        stats_text += f"Theory integral: {np.trapezoid(y_theory, x_theory):.2f}"
    
    plt.gcf().text(0.72, 0.15, stats_text, 
                  bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'dos_comparison_λ{λ}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3D Visualization of wavefunction (unchanged)
def plot_3d_wavefunction(states, energies, λ, filename):
    """Create a 3D plot of wavefunction amplitudes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(states.shape[0])
    
    # Plot a subset of states for clarity
    step = max(1, states.shape[1] // 10)
    for idx in range(0, states.shape[1], step):
        y = np.ones_like(x) * energies[idx]
        z = np.abs(states[:, idx])**2
        ax.plot(x, y, z, color=cmap(idx/states.shape[1]), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Site n')
    ax.set_ylabel('Energy E')
    ax.set_zlabel('|ψₙ|²')
    ax.set_title(f'3D Wavefunction Amplitudes (λ={λ})')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_combined_lyapunov(evals_all_λ, le_all_λ, λ_vals):
    """Plot all Lyapunov exponents on the same figure."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(λ_vals)))
    
    for idx, λ in enumerate(λ_vals):
        evals_avg = np.mean(evals_all_λ[idx], axis=0)
        le_avg = np.mean(le_all_λ[idx], axis=0)
        
        if λ == 0:
            plt.axhline(0, color=colors[idx], linestyle='--', 
                       label=f'λ={λ} (Ballistic)')
        else:
            plt.scatter(evals_avg, le_avg, s=20, color=colors[idx],
                       alpha=0.7, label=f'λ={λ}')
    
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent γ')
    plt.title('Lyapunov Exponents for Different Potential Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('combined_lyapunov.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_ipr(evals_all_λ, iprs_all_λ, λ_vals):
    """Plot all IPRs on the same figure."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(λ_vals)))
    
    for idx, λ in enumerate(λ_vals):
        evals_avg = np.mean(evals_all_λ[idx], axis=0)
        iprs_avg = np.mean(iprs_all_λ[idx], axis=0)
        
        plt.scatter(evals_avg, iprs_avg, s=20, color=colors[idx],
                   alpha=0.7, label=f'λ={λ}')
    
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.yscale('log')
    plt.title('IPR for Different Potential Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('combined_ipr.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_time_evolution_animation(H, N, λ, num_frames=100, duration=10):
    """Create and save animation of wavefunction time evolution."""
    # Initial wavefunction localized at center
    psi0 = np.zeros(N)
    psi0[N//2] = 1.0
    
    # Time points for animation
    times = np.linspace(0, duration, num_frames)
    prob_t = time_evolution(H, psi0, times)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=2)
    
    # Set axis limits
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, np.max(prob_t)*1.1)
    ax.set_xlabel('Site n')
    ax.set_ylabel('Probability |ψₙ|²')
    ax.set_title(f'Time Evolution (λ={λ})')
    ax.grid(True, alpha=0.3)
    
    # Initialization function
    def init():
        line.set_data([], [])
        return line,
    
    # Animation function
    def animate(i):
        line.set_data(np.arange(N), prob_t[i])
        ax.set_title(f'Time Evolution (λ={λ}, t={times[i]:.2f})')
        return line,
    
    # Create animation
    ani = FuncAnimation(fig, animate, frames=num_frames,
                        init_func=init, blit=True, interval=50)
    
    # Save animation
    writer = PillowWriter(fps=20)
    ani.save(f'time_evolution_λ{λ}.gif', writer=writer)
    plt.close()

# Main simulation
def run_simulation(N=200, t=1.0, λ_vals=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10], num_realizations=5):
    colors = sns.color_palette("viridis", len(λ_vals))
    iprs_all_λ = []
    evals_all_λ = []
    le_all_λ = []
    pr_all_λ = []
    xi_numerical = []
    xi_theoretical = []
    
    for λ_idx, λ in enumerate(λ_vals):
        print(f"\n--- Simulation for λ = {λ} ---")
        iprs_all = np.zeros((num_realizations, N))
        evals_all = np.zeros((num_realizations, N))
        le_all = np.zeros((num_realizations, N))
        pr_all = np.zeros((num_realizations, N))
        evecs_last = None
        H_last = None
        
        # Compute over realizations (different phases φ)
        for r in range(num_realizations):
            sys.stdout.write(f'\rComputing λ={λ}, realization {r+1}/{num_realizations}')
            sys.stdout.flush()
            φ = np.random.uniform(0, 2*np.pi)  # Random phase for each realization
            H, potential = maryland_hamiltonian(N, λ, φ, t)
            evals, evecs = eigh(H)
            iprs_all[r, :] = compute_ipr(evecs)
            le_all[r, :] = compute_lyapunov(evecs, λ)
            pr_all[r, :] = compute_pr(evecs)
            evals_all[r, :] = evals
            if r == num_realizations - 1:
                evecs_last = evecs
                H_last = H
        sys.stdout.write('\n')
        
        # Store results for comparison plots
        iprs_all_λ.append(iprs_all)
        evals_all_λ.append(evals_all)
        le_all_λ.append(le_all)
        pr_all_λ.append(pr_all)
        
        # Average results
        evals_avg = np.mean(evals_all, axis=0)
        iprs_avg = np.mean(iprs_all, axis=0)
        le_avg = np.mean(le_all, axis=0)
        pr_avg = np.mean(pr_all, axis=0)
        
        # Plot potential profile
        plt.figure(figsize=(12, 6))
        plt.plot(potential, color=colors[λ_idx], linewidth=2, alpha=0.8)
        plt.fill_between(range(N), potential, color=colors[λ_idx], alpha=0.2)
        plt.xlabel('Site n')
        plt.ylabel('Potential Vₙ')
        plt.title(f'Maryland Potential Profile (λ={λ})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'potential_λ{λ}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # IPR vs Energy plot
        plt.figure(figsize=(12, 8))
        sc = plt.scatter(evals_avg, iprs_avg, s=50, c=pr_avg, cmap='viridis', alpha=0.8, 
                        vmin=0, vmax=N/2, edgecolors='k', linewidths=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label('Participation Ratio', rotation=270, labelpad=20)
        plt.xlabel('Energy E')
        plt.ylabel('IPR')
        plt.yscale('log')
        plt.title(f'IPR vs Energy (λ={λ})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'ipr_λ{λ}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Lyapunov Exponent plot
        plt.figure(figsize=(12, 8))
        if λ == 0:
            plt.text(0.5, 0.5, 'γ → 0 (Extended)', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=20, bbox=dict(facecolor='yellow', alpha=0.5))
        else:
            sc = plt.scatter(evals_avg, le_avg, s=50, c=iprs_avg, cmap='plasma', alpha=0.8,
                           vmin=0, vmax=0.1, edgecolors='k', linewidths=0.5)
            cbar = plt.colorbar(sc)
            cbar.set_label('IPR', rotation=270, labelpad=20)
        
        plt.xlabel('Energy E')
        plt.ylabel('Lyapunov Exponent γ')
        plt.title(f'Lyapunov Exponent (λ={λ})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'lyapunov_λ{λ}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Level spacing statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        level_spacing_statistics(evals_avg, ax, label=f'λ={λ}', color=colors[λ_idx])
        plt.title(f'Level Spacing Distribution (λ={λ})')
        plt.savefig(f'level_spacing_λ{λ}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DOS comparison
        plot_dos_comparison(evals_all, λ, t)
        
        # Localization length analysis (only for λ ≠ 0)
        if λ != 0:
            idx_most_localized = np.argmax(iprs_avg)
            xi_fit, (x, prob, fit_curve) = fit_localization_length(evecs_last[:, idx_most_localized])
        else:
            xi_fit = float('inf')
            x = np.arange(N)
            prob = np.abs(evecs_last[:, N//2])**2
            fit_curve = np.zeros_like(x)
        
        xi_theory = theoretical_localization_length(λ, t)
        xi_numerical.append(xi_fit)
        xi_theoretical.append(xi_theory)
        
        # Localization length fit plot
        plt.figure(figsize=(12, 8))
        plt.scatter(x, prob, s=50, color=colors[λ_idx], alpha=0.6, 
                   label='Numerical |ψ(x)|²')
        if λ != 0:
            plt.plot(x, fit_curve, 'k-', linewidth=3, alpha=0.8,
                    label=f'Fit: ξ = {xi_fit:.2f}')
        plt.axhline(0, color='gray', lw=0.5)
        plt.yscale('log')
        plt.xlabel('Site x')
        plt.ylabel('|ψ(x)|²')
        title = f'Localization Fit (λ={λ})'
        if λ == 0:
            title += ', ξ → ∞ (Extended)'
        else:
            title += f', ξ_theory = {xi_theory:.2f}'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'loc_length_fit_λ{λ}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3D Wavefunction plot
        plot_3d_wavefunction(evecs_last, evals_avg, λ, f'3d_wavefunction_λ{λ}.png')
        
        # Create time evolution animation
        create_time_evolution_animation(H_last, N, λ)
        
        # Print results
        print(f"  Fitted localization length ξ ≈ {xi_fit if λ != 0 else 'inf'}")
        print(f"  Theoretical ξ ≈ {xi_theory if λ != 0 else 'inf'}")
    
    # Create combined comparison plots
    plot_combined_lyapunov(evals_all_λ, le_all_λ, λ_vals)
    plot_combined_ipr(evals_all_λ, iprs_all_λ, λ_vals)
    
    # Plot localization length vs λ
    plt.figure(figsize=(12, 8))
    plt.semilogy(λ_vals, xi_numerical, 'o-', label='Numerical ξ', markersize=8)
    plt.semilogy(λ_vals, xi_theoretical, 's--', label='Theoretical ξ', markersize=8)
    plt.xlabel('Potential Strength λ')
    plt.ylabel('Localization Length ξ')
    plt.title('Localization Length vs Potential Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loc_length_vs_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run simulation
if __name__ == "__main__":
    run_simulation()