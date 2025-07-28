import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.special import jv
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
import os
from tqdm import tqdm

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

# Create directory structure
def create_directories():
    base_dir = "stark_anderson_results"
    subdirs = ["IPR", "Lyapunov", "Bessel", "DOS", "LevelSpacing", "Wavefunction", 
               "Disorder", "TimeEvolution", "LocalizationLength", "WannierStark", "IPRDistribution"]
    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    return base_dir

# Define the Stark-Anderson Hamiltonian
def stark_anderson_hamiltonian(N, W, t=1.0, F=0.6):
    """Construct the 1D Stark-Anderson Hamiltonian with random disorder and electric field."""
    disorder = np.random.uniform(-W/2, W/2, size=N)
    stark = F * np.arange(N)
    diagonal = disorder + stark
    off_diagonal = -t * np.ones(N-1)
    H = diags([off_diagonal, diagonal, off_diagonal], offsets=[-1, 0, 1]).toarray()
    return H, disorder

# IPR calculation
def compute_ipr(eigenstates):
    """Calculate Inverse Participation Ratio (IPR)."""
    psi_abs = np.abs(eigenstates)
    psi_norm = np.sum(psi_abs**2, axis=0)
    mask = psi_norm > 1e-10
    psi_normalized = psi_abs / np.sqrt(psi_norm + 1e-10)
    ipr = np.sum(psi_normalized**4, axis=0)
    ipr[~mask] = 0.0
    return ipr

# Participation Ratio calculation
def compute_pr(eigenstates):
    """Calculate Participation Ratio (PR)."""
    ipr = compute_ipr(eigenstates)
    pr = 1 / (ipr + 1e-10)
    return pr

# Lyapunov Exponent calculation
def compute_lyapunov(eigenstates, W, F):
    """Calculate Lyapunov exponent via wavefunction decay."""
    if W == 0 and F == 0:
        return np.zeros(eigenstates.shape[1])
    le = []
    for psi in eigenstates.T:
        psi = psi / np.linalg.norm(psi)
        log_psi = np.log(np.abs(psi) + 1e-10)
        x = np.arange(len(psi))
        slope, _ = np.polyfit(x, log_psi, 1)
        le.append(-slope)
    return np.array(le)

# Level spacing statistics
def level_spacing_statistics(energies, ax, label, color):
    """Plot level spacing distribution with theoretical comparisons."""
    sorted_E = np.sort(energies)
    s = np.diff(sorted_E)
    if len(s) > 1:
        s_mean = s / np.mean(s)
        sns.histplot(s_mean, bins=50, stat='density', alpha=0.6, color=color, label=label, ax=ax)
        x = np.linspace(0, 3, 200)
        ax.plot(x, np.exp(-x), 'k--', label='Poisson (Localized)', linewidth=2)
        ax.plot(x, (np.pi/2)*x*np.exp(-np.pi*x**2/4), 'r--', label='Wigner-Dyson (Extended)', linewidth=2)
        ax.set_xlabel("Normalized Level Spacing, s")
        ax.set_ylabel("Probability Density, P(s)")
        ax.legend()

# Localization length fit
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

# Theoretical localization length
def theoretical_localization_length(W, t=1.0, F=0.6):
    """Calculate theoretical localization length, considering Stark effect."""
    if W == 0 and F == 0:
        return float('inf')
    xi_anderson = 96 * t**2 / (W**2 + 1e-10) if W != 0 else float('inf')
    xi_stark = 2 * t / (F + 1e-10) if F != 0 else float('inf')
    return 1 / (1/xi_anderson + 1/xi_stark)

# Time evolution
def time_evolution(H, psi0, times):
    """Compute time evolution of wavefunction."""
    from scipy.linalg import expm
    psi_t = []
    for t_val in times:
        U = expm(-1j * H * t_val)
        psi_t.append(np.abs(U @ psi0)**2)
    return np.array(psi_t)

# Density of States calculation
def compute_dos(energies, sigma=0.1):
    """Compute density of states using Gaussian kernel density estimation."""
    kde = gaussian_kde(energies, bw_method=sigma)
    x = np.linspace(np.min(energies), np.max(energies), 500)
    y = kde(x)
    return x, y

# Theoretical DOS for clean system
def theoretical_dos(E, t=1.0):
    """Theoretical density of states for clean 1D system without electric field."""
    return 1 / (np.pi * np.sqrt(4 * t**2 - E**2 + 1e-10))

def plot_dos_comparison(evals, W, F, t=1.0, base_dir="stark_anderson_results"):
    """Create enhanced DOS comparison plot."""
    plt.figure(figsize=(12, 8))
    e_min, e_max = np.min(evals), np.max(evals)
    energy_range = e_max - e_min
    n_bins = 200
    hist, bin_edges = np.histogram(evals, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = bin_edges[1] - bin_edges[0]
    
    plt.bar(bin_centers, hist, width=bin_width*0.9, color='blue', alpha=0.7, edgecolor='black', 
            linewidth=0.5, label=f'Numerical Histogram ({n_bins} bins)')
    x_kde, y_kde = compute_dos(evals.flatten())
    plt.plot(x_kde, y_kde, 'red', linewidth=2, alpha=0.9, label='Numerical KDE')
    
    if W == 0 and F == 0:
        x_theory = np.linspace(-2*t, 2*t, 200)
        y_theory = theoretical_dos(x_theory, t)
        plt.plot(x_theory, y_theory, 'green', linewidth=3, alpha=0.8, label='Theoretical DOS')
    
    plt.xlabel('Energy E')
    plt.ylabel('Density of States')
    plt.title(f'High-Resolution DOS Comparison (W={W}, F={F})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    stats_text = f"""
    Statistics (W={W}, F={F}):
    Mean energy: {np.mean(evals):.2f}
    Energy std: {np.std(evals):.2f}
    Histogram integral: {np.sum(hist)*bin_width:.2f}
    KDE integral: {np.trapezoid(y_kde, x_kde):.2f}
    """
    if W == 0 and F == 0:
        stats_text += f"Theory integral: {np.trapezoid(y_theory, x_theory):.2f}"
    
    plt.gcf().text(0.72, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "DOS", f'dos_comparison_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_wavefunction(states, energies, W, F, base_dir):
    """Create a 3D plot of wavefunction amplitudes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(states.shape[0])
    step = max(1, states.shape[1] // 10)
    for idx in range(0, states.shape[1], step):
        y = np.ones_like(x) * energies[idx]
        z = np.abs(states[:, idx])**2
        ax.plot(x, y, z, color=cmap(idx/states.shape[1]), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Site n')
    ax.set_ylabel('Energy E')
    ax.set_zlabel('|ψₙ|²')
    ax.set_title(f'3D Wavefunction Amplitudes (W={W}, F={F})')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Wavefunction", f'3d_wavefunction_W{W}_F{F}.png'), dpi=300)
    plt.close()

def plot_bessel_functions(F, base_dir):
    """Plot Bessel functions relevant to Stark effect."""
    plt.figure(figsize=(12, 8))
    x = np.linspace(-10, 10, 1000)
    for n in range(3):
        plt.plot(x, jv(n, x/F), label=f'J_{n}(x/F)', linewidth=2)
    plt.xlabel('x/F')
    plt.ylabel('J_n(x/F)')
    plt.title(f'Bessel Functions for Stark Effect (F={F})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "Bessel", f'bessel_functions_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_wannier_stark_ladder(evals, W, F, base_dir):
    """Plot energy spectrum to highlight Wannier-Stark ladder."""
    plt.figure(figsize=(12, 8))
    sorted_evals = np.sort(evals)
    plt.plot(range(len(sorted_evals)), sorted_evals, 'o-', color='blue', alpha=0.7)
    plt.xlabel('Eigenstate Index')
    plt.ylabel('Energy E')
    plt.title(f'Energy Spectrum (Wannier-Stark Ladder) (W={W}, F={F})')
    if W == 0 and F != 0:
        # Expected spacing is F for Wannier-Stark ladder
        plt.axhline(y=sorted_evals[0], color='gray', linestyle='--', alpha=0.3)
        for i in range(1, len(sorted_evals)):
            plt.axhline(y=sorted_evals[0] + i*F, color='gray', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "WannierStark", f'wannier_stark_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ipr_distribution(iprs, W, F, base_dir):
    """Plot histogram of IPR values."""
    plt.figure(figsize=(12, 8))
    sns.histplot(iprs.flatten(), bins=50, stat='density', alpha=0.6, color='purple')
    plt.xlabel('IPR')
    plt.ylabel('Density')
    plt.title(f'IPR Distribution (W={W}, F={F})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "IPRDistribution", f'ipr_distribution_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_lyapunov(evals_all_W, le_all_W, W_vals, F, base_dir):
    """Plot all Lyapunov exponents on the same figure."""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(W_vals)))
    
    for idx, W in enumerate(W_vals):
        evals_avg = np.mean(evals_all_W[idx], axis=0)
        le_avg = np.mean(le_all_W[idx], axis=0)
        if W == 0 and F == 0:
            plt.axhline(0, color=colors[idx], linestyle='--', label=f'W={W}, F={F} (Ballistic)')
        else:
            plt.scatter(evals_avg, le_avg, s=20, color=colors[idx], alpha=0.7, label=f'W={W}, F={F}')
    
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent γ')
    plt.title(f'Lyapunov Exponents for Different Disorder Strengths (F={F})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "Lyapunov", f'combined_lyapunov_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_ipr(evals_all_W, iprs_all_W, W_vals, F, base_dir):
    """Plot all IPRs on the same figure."""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(W_vals)))
    
    for idx, W in enumerate(W_vals):
        evals_avg = np.mean(evals_all_W[idx], axis=0)
        iprs_avg = np.mean(iprs_all_W[idx], axis=0)
        plt.scatter(evals_avg, iprs_avg, s=20, color=colors[idx], alpha=0.7, label=f'W={W}, F={F}')
    
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.yscale('log')
    plt.title(f'IPR for Different Disorder Strengths (F={F})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "IPR", f'combined_ipr_F{F}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_localization_length_map(xi_numerical_all, xi_theoretical_all, W_vals, F_vals, base_dir):
    """Plot localization length as a function of W and F."""
    plt.figure(figsize=(12, 8))
    W_mesh, F_mesh = np.meshgrid(W_vals, F_vals)
    xi_numerical_array = np.array(xi_numerical_all).reshape(len(F_vals), len(W_vals))
    xi_theoretical_array = np.array(xi_theoretical_all).reshape(len(F_vals), len(W_vals))
    
    # Numerical localization length
    plt.subplot(1, 2, 1)
    c = plt.contourf(W_mesh, F_mesh, xi_numerical_array, cmap='viridis', levels=20)
    plt.colorbar(c, label='Numerical ξ')
    plt.xlabel('Disorder Strength W')
    plt.ylabel('Electric Field F')
    plt.title('Numerical Localization Length')
    plt.xscale('log')
    
    # Theoretical localization length
    plt.subplot(1, 2, 2)
    c = plt.contourf(W_mesh, F_mesh, xi_theoretical_array, cmap='viridis', levels=20)
    plt.colorbar(c, label='Theoretical ξ')
    plt.xlabel('Disorder Strength W')
    plt.ylabel('Electric Field F')
    plt.title('Theoretical Localization Length')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "LocalizationLength", 'localization_length_map.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_time_evolution_animation(H, N, W, F, base_dir, num_frames=100, duration=10):
    """Create and save animation of wavefunction time evolution."""
    psi0 = np.zeros(N)
    psi0[N//2] = 1.0
    times = np.linspace(0, duration, num_frames)
    prob_t = time_evolution(H, psi0, times)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, np.max(prob_t)*1.1)
    ax.set_xlabel('Site n')
    ax.set_ylabel('Probability |ψₙ|²')
    ax.set_title(f'Time Evolution (W={W}, F={F})')
    ax.grid(True, alpha=0.3)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        line.set_data(np.arange(N), prob_t[i])
        ax.set_title(f'Time Evolution (W={W}, F={F}, t={times[i]:.2f})')
        return line,
    
    ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True, interval=50)
    writer = PillowWriter(fps=20)
    ani.save(os.path.join(base_dir, "TimeEvolution", f'time_evolution_W{W}_F{F}.gif'), writer=writer)
    plt.close()

# Main simulation
def run_simulation(N=200, t=1.0, F_vals=[0.6, 0.7, 0.8, 0.9, 1.0], 
                   W_vals=[0, 0.5, 1, 2, 5, 10, 20, 50, 100, 10000], num_realizations=5):
    base_dir = create_directories()
    xi_numerical_all = []
    xi_theoretical_all = []
    
    for F in F_vals:
        print(f"\n=== Simulation for Electric Field F = {F} ===")
        # Plot Bessel functions once per F
        plot_bessel_functions(F, base_dir)
        
        iprs_all_W = []
        evals_all_W = []
        le_all_W = []
        pr_all_W = []
        xi_numerical = []
        xi_theoretical = []
        
        for W_idx, W in enumerate(tqdm(W_vals, desc=f"Processing W for F={F}")):
            print(f"\n--- Simulation for W = {W}, F = {F} ---")
            iprs_all = np.zeros((num_realizations, N))
            evals_all = np.zeros((num_realizations, N))
            le_all = np.zeros((num_realizations, N))
            pr_all = np.zeros((num_realizations, N))
            evecs_last = None
            H_last = None
            
            for r in range(num_realizations):
                sys.stdout.write(f'\rComputing W={W}, F={F}, realization {r+1}/{num_realizations}')
                sys.stdout.flush()
                H, disorder = stark_anderson_hamiltonian(N, W, t, F)
                evals, evecs = eigh(H)
                iprs_all[r, :] = compute_ipr(evecs)
                le_all[r, :] = compute_lyapunov(evecs, W, F)
                pr_all[r, :] = compute_pr(evecs)
                evals_all[r, :] = evals
                if r == num_realizations - 1:
                    evecs_last = evecs
                    H_last = H
            sys.stdout.write('\n')
            
            iprs_all_W.append(iprs_all)
            evals_all_W.append(evals_all)
            le_all_W.append(le_all)
            pr_all_W.append(pr_all)
            
            evals_avg = np.mean(evals_all, axis=0)
            iprs_avg = np.mean(iprs_all, axis=0)
            le_avg = np.mean(le_all, axis=0)
            pr_avg = np.mean(pr_all, axis=0)
            
            # Disorder profile
            plt.figure(figsize=(12, 6))
            plt.plot(disorder, color=cmap(W_idx/len(W_vals)), linewidth=2, alpha=0.8)
            plt.fill_between(range(N), disorder, color=cmap(W_idx/len(W_vals)), alpha=0.2)
            plt.xlabel('Site n')
            plt.ylabel('Disorder + Stark Potential Vₙ')
            plt.title(f'Disorder Profile with Stark Effect (W={W}, F={F})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(base_dir, "Disorder", f'disorder_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
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
            plt.title(f'IPR vs Energy (W={W}, F={F})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(base_dir, "IPR", f'ipr_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Lyapunov Exponent plot
            plt.figure(figsize=(12, 8))
            if W == 0 and F == 0:
                plt.text(0.5, 0.5, 'γ → 0 (Ballistic)', 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=20, bbox=dict(facecolor='yellow', alpha=0.5))
            else:
                sc = plt.scatter(evals_avg, le_avg, s=50, c=iprs_avg, cmap='plasma', alpha=0.8,
                               vmin=0, vmax=0.1, edgecolors='k', linewidths=0.5)
                cbar = plt.colorbar(sc)
                cbar.set_label('IPR', rotation=270, labelpad=20)
            
            plt.xlabel('Energy E')
            plt.ylabel('Lyapunov Exponent γ')
            plt.title(f'Lyapunov Exponent (W={W}, F={F})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(base_dir, "Lyapunov", f'lyapunov_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Level spacing statistics
            fig, ax = plt.subplots(figsize=(10, 6))
            level_spacing_statistics(evals_avg, ax, label=f'W={W}, F={F}', color=cmap(W_idx/len(W_vals)))
            plt.title(f'Level Spacing Distribution (W={W}, F={F})')
            plt.savefig(os.path.join(base_dir, "LevelSpacing", f'level_spacing_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # DOS comparison
            plot_dos_comparison(evals_all, W, F, t, base_dir)
            
            # Wannier-Stark ladder
            plot_wannier_stark_ladder(evals_avg, W, F, base_dir)
            
            # IPR distribution
            plot_ipr_distribution(iprs_all, W, F, base_dir)
            
            # Localization length analysis
            idx_most_localized = np.argmax(iprs_avg)
            xi_fit, (x, prob, fit_curve) = fit_localization_length(evecs_last[:, idx_most_localized])
            xi_theory = theoretical_localization_length(W, t, F)
            xi_numerical.append(xi_fit)
            xi_theoretical.append(xi_theory)
            
            # Localization length fit plot
            plt.figure(figsize=(12, 8))
            plt.scatter(x, prob, s=50, color=cmap(W_idx/len(W_vals)), alpha=0.6, label='Numerical |ψ(x)|²')
            plt.plot(x, fit_curve, 'k-', linewidth=3, alpha=0.8, label=f'Fit: ξ = {xi_fit:.2f}')
            plt.axhline(0, color='gray', lw=0.5)
            plt.yscale('log')
            plt.xlabel('Site x')
            plt.ylabel('|ψ(x)|²')
            title = f'Localization Fit (W={W}, F={F})'
            if W == 0 and F == 0:
                title += ', ξ → ∞'
            else:
                title += f', ξ_theory = {xi_theory:.2f}'
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(base_dir, "LocalizationLength", f'loc_length_fit_W{W}_F{F}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3D Wavefunction plot
            plot_3d_wavefunction(evecs_last, evals_avg, W, F, base_dir)
            
            # Time evolution animation
            create_time_evolution_animation(H_last, N, W, F, base_dir)
            
            # Print results
            print(f"  Fitted localization length ξ ≈ {xi_fit:.2f}")
            print(f"  Theoretical ξ ≈ {xi_theory if (W != 0 or F != 0) else 'inf'}")
        
        xi_numerical_all.append(xi_numerical)
        xi_theoretical_all.append(xi_theoretical)
        
        # Combined comparison plots for this F
        plot_combined_lyapunov(evals_all_W, le_all_W, W_vals, F, base_dir)
        plot_combined_ipr(evals_all_W, iprs_all_W, W_vals, F, base_dir)
    
    # Localization length map
    plot_localization_length_map(xi_numerical_all, xi_theoretical_all, W_vals, F_vals, base_dir)

# Run simulation
if __name__ == "__main__":
    run_simulation()
