import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
import sys
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set styles
sns.set(style="whitegrid", palette="viridis")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.figsize': (14, 8),
    'axes.grid': True,
    'grid.alpha': 0.3
})
cmap = plt.cm.viridis

# Create output directories
def create_directories(W_vals):
    """Create folder structure for output files."""
    base_dir = "anderson_output"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "disorder_profiles",
        "dos",
        "wavefunctions",
        "ipr",
        "lyapunov",
        "comparisons",
        "level_spacing",
        "time_evolution",
        "connection"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir

# ===================================================================
# Model Implementations
# ===================================================================

def harper_hofstadter_hamiltonian(Nx, Ny, alpha, t=1.0, W=0):
    """Construct 2D Harper-Hofstadter Hamiltonian with magnetic field and disorder."""
    total_sites = Nx * Ny
    H = np.zeros((total_sites, total_sites), dtype=complex)
    
    # On-site disorder
    disorder = np.random.uniform(-W/2, W/2, total_sites)
    np.fill_diagonal(H, disorder)
    
    # Hopping terms with Peierls phases
    for m in range(Ny):
        for n in range(Nx):
            idx = m * Nx + n
            
            # Hopping in x-direction (no phase)
            if n < Nx - 1:
                idx_right = m * Nx + (n + 1)
                H[idx, idx_right] = -t
                H[idx_right, idx] = -t
                
            # Hopping in y-direction (with magnetic phase)
            if m < Ny - 1:
                idx_up = (m + 1) * Nx + n
                phase = 2j * np.pi * alpha * n
                H[idx, idx_up] = -t * np.exp(phase)
                H[idx_up, idx] = -t * np.exp(-phase)
    return H, disorder

def aubry_andre_hamiltonian(N, lambda_, beta, phi=0, W=0, t=1.0):
    """Construct 1D Aubry-André Hamiltonian with quasiperiodic potential."""
    quasi_pot = lambda_ * np.cos(2 * np.pi * beta * np.arange(N) + phi)
    disorder_pot = np.random.uniform(-W/2, W/2, N) if W > 0 else np.zeros(N)
    diagonal = quasi_pot + disorder_pot
    
    off_diagonal = -t * np.ones(N - 1)
    H = diags([off_diagonal, diagonal, off_diagonal], 
              offsets=[-1, 0, 1]).toarray()
    return H, diagonal

# ===================================================================
# Analysis Functions
# ===================================================================

def compute_ipr(eigenstates):
    """Calculate Inverse Participation Ratio (IPR) with normalization check."""
    # Check normalization of eigenstates
    norms = np.sum(np.abs(eigenstates)**2, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-5):
        # Normalize if not already normalized
        eigenstates = eigenstates / np.sqrt(norms)
    
    psi_abs = np.abs(eigenstates)
    ipr = np.sum(psi_abs**4, axis=0)
    return ipr

def compute_lyapunov(eigenstates, W):
    """Calculate Lyapunov exponent via wavefunction decay."""
    if W == 0:
        return np.zeros(eigenstates.shape[1])
    
    le = []
    for psi in eigenstates.T:
        psi = psi / np.linalg.norm(psi)  # Ensure normalization
        log_psi = np.log(np.abs(psi) + 1e-10)
        x = np.arange(len(psi))
        slope, _ = np.polyfit(x, log_psi, 1)
        le.append(-slope)
    return np.array(le)

def level_spacing_statistics(energies):
    """Calculate level spacing distribution."""
    sorted_E = np.sort(energies)
    s = np.diff(sorted_E)
    if len(s) > 1:
        s_mean = np.mean(s)
        s_normalized = s / s_mean
        return s_normalized
    return np.array([])

def compute_dos(energies, sigma=0.1):
    """Compute density of states using Gaussian kernel density estimation."""
    kde = gaussian_kde(energies, bw_method=sigma)
    x = np.linspace(np.min(energies), np.max(energies), 500)
    y = kde(x)
    return x, y

def plot_3d_wavefunction(states, energies, W, filename):
    """Create a 3D plot of wavefunction amplitudes with normalization check."""
    # Verify normalization
    norms = np.sum(np.abs(states)**2, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-5):
        states = states / np.sqrt(norms)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(states.shape[0])
    
    # Plot a subset of states
    step = max(1, states.shape[1] // 10)
    for idx in range(0, states.shape[1], step):
        y = np.ones_like(x) * energies[idx]
        z = np.abs(states[:, idx])**2
        ax.plot(x, y, z, color=cmap(idx/states.shape[1]), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Site Index')
    ax.set_ylabel('Energy E')
    ax.set_zlabel('|ψₙ|²')
    ax.set_title(f'Harper-Hofstadter Wavefunctions (W={W})')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def time_evolution(H, psi0, times):
    """Compute time evolution of wavefunction with normalization."""
    from scipy.linalg import expm
    psi_t = []
    for t_val in times:
        U = expm(-1j * H * t_val)
        psi = U @ psi0
        # Normalize at each time step
        psi = psi / np.linalg.norm(psi)
        psi_t.append(np.abs(psi)**2)
    return np.array(psi_t)

def create_time_evolution_animation(H, N, W, base_dir, num_frames=100, duration=10):
    """Create and save animation of wavefunction time evolution."""
    # Initial wavefunction localized at center
    psi0 = np.zeros(N, dtype=complex)
    center = N // 2
    psi0[center] = 1.0
    psi0 = psi0 / np.linalg.norm(psi0)  # Normalize
    
    # Time points for animation
    times = np.linspace(0, duration, num_frames)
    prob_t = time_evolution(H, psi0, times)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=2)
    
    # Set axis limits
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, np.max(prob_t)*1.1)
    ax.set_xlabel('Site Index')
    ax.set_ylabel('Probability |ψₙ|²')
    ax.set_title(f'Time Evolution (W={W})')
    ax.grid(True, alpha=0.3)
    
    # Initialization function
    def init():
        line.set_data([], [])
        return line,
    
    # Animation function
    def animate(i):
        line.set_data(np.arange(N), prob_t[i])
        ax.set_title(f'Time Evolution (W={W}, t={times[i]:.2f})')
        return line,
    
    # Create animation
    ani = FuncAnimation(fig, animate, frames=num_frames,
                        init_func=init, blit=True, interval=50)
    
    # Save animation
    writer = PillowWriter(fps=20)
    anim_path = os.path.join(base_dir, "time_evolution", f'time_evolution_W{W}.gif')
    ani.save(anim_path, writer=writer)
    plt.close()
    print(f"Saved time evolution animation for W={W}")

# ===================================================================
# Visualization Functions
# ===================================================================

def plot_hofstadter_butterfly(base_dir, Nx=30, Ny=30, t=1.0, alpha_min=0, alpha_max=1, num_alphas=200):
    """Compute and plot the Hofstadter butterfly spectrum."""
    alphas = np.linspace(alpha_min, alpha_max, num_alphas)
    all_energies = []
    
    print("Generating Hofstadter butterfly...")
    for i, alpha in enumerate(alphas):
        sys.stdout.write(f'\rProgress: {100*(i+1)/len(alphas):.1f}%')
        sys.stdout.flush()
        H, _ = harper_hofstadter_hamiltonian(Nx, Ny, alpha, t, W=0)
        evals = np.linalg.eigvalsh(H)
        all_energies.append(evals)
    
    plt.figure(figsize=(14, 8))
    for i, alpha in enumerate(alphas):
        plt.scatter([alpha] * len(all_energies[i]), all_energies[i], 
                   s=0.5, color='blue', alpha=0.6)
    
    plt.xlabel('Flux per plaquette (α)')
    plt.ylabel('Energy E')
    plt.title('Hofstadter Butterfly Spectrum')
    plt.grid(True, alpha=0.3)
    butterfly_path = os.path.join(base_dir, "hofstadter_butterfly.png")
    plt.savefig(butterfly_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("\nHofstadter butterfly saved.")

def plot_dos(evals, W, base_dir, model_name="Harper-Hofstadter"):
    """Plot DOS with both KDE and histogram bars."""
    plt.figure(figsize=(14, 8))
    
    # Compute KDE
    x_kde, y_kde = compute_dos(evals, sigma=0.1)
    plt.plot(x_kde, y_kde, 'b-', linewidth=3, alpha=0.8, label='KDE Estimate')
    
    # Compute histogram
    hist, bin_edges = np.histogram(evals, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_centers, hist, width=bin_width*0.8, 
            color='orange', alpha=0.6, edgecolor='black', 
            label='Histogram')
    
    plt.xlabel('Energy E')
    plt.ylabel('Density of States')
    plt.title(f'{model_name} DOS (W={W})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to dos subdirectory
    dos_path = os.path.join(base_dir, "dos", f'dos_W{W}.png')
    plt.savefig(dos_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_level_spacing(energies, W, base_dir, model_name="Harper-Hofstadter"):
    """Plot level spacing distribution."""
    spacings = level_spacing_statistics(energies)
    if len(spacings) == 0:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Theoretical distributions
    x = np.linspace(0, 3, 200)
    poisson = np.exp(-x)
    wigner = (np.pi/2) * x * np.exp(-np.pi*x**2/4)
    
    # Plot histogram
    plt.hist(spacings, bins=50, density=True, alpha=0.6, 
             color='blue', label='Numerical')
    plt.plot(x, poisson, 'k--', linewidth=2, label='Poisson (Localized)')
    plt.plot(x, wigner, 'r--', linewidth=2, label='Wigner-Dyson (Extended)')
    
    plt.xlabel("Normalized Level Spacing, s")
    plt.ylabel("Probability Density, P(s)")
    plt.title(f'{model_name} Level Spacing (W={W})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to level_spacing subdirectory
    level_path = os.path.join(base_dir, "level_spacing", f'level_spacing_W{W}.png')
    plt.savefig(level_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_disorder_profile(disorder, W, base_dir, model_name="Harper-Hofstadter"):
    """Plot disorder potential profile."""
    plt.figure(figsize=(12, 6))
    plt.plot(disorder, 'b-', linewidth=1.5, alpha=0.8)
    plt.fill_between(range(len(disorder)), disorder, color='b', alpha=0.2)
    plt.xlabel('Site Index')
    plt.ylabel('Disorder Potential')
    plt.title(f'{model_name} Disorder Profile (W={W})')
    plt.grid(True, alpha=0.3)
    
    # Save to disorder_profiles subdirectory
    disorder_path = os.path.join(base_dir, "disorder_profiles", f'disorder_W{W}.png')
    plt.savefig(disorder_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_wavefunction_analysis(evecs, evals, W, base_dir):
    """Plot IPR and Lyapunov for Harper-Hofstadter."""
    # Compute metrics
    ipr = compute_ipr(evecs)
    le = compute_lyapunov(evecs, W)
    
    # IPR plot
    plt.figure(figsize=(12, 8))
    plt.scatter(evals, ipr, s=20, c=le, cmap='viridis', alpha=0.7)
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.yscale('log')
    plt.title(f'Harper-Hofstadter: IPR vs Energy (W={W})')
    plt.colorbar(label='Lyapunov Exponent')
    plt.grid(True, alpha=0.3)
    ipr_path = os.path.join(base_dir, "ipr", f'ipr_W{W}.png')
    plt.savefig(ipr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Lyapunov plot
    plt.figure(figsize=(12, 8))
    plt.scatter(evals, le, s=20, c=ipr, cmap='plasma', alpha=0.7)
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent')
    plt.title(f'Harper-Hofstadter: Lyapunov Exponent vs Energy (W={W})')
    plt.colorbar(label='IPR')
    plt.grid(True, alpha=0.3)
    lyap_path = os.path.join(base_dir, "lyapunov", f'lyapunov_W{W}.png')
    plt.savefig(lyap_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_comparisons(hh_results, aa_results, base_dir):
    """Create comprehensive comparison plots across all W values."""
    # Prepare data
    all_hh_evals = []
    all_hh_ipr = []
    all_hh_lyap = []
    all_aa_evals = []
    all_aa_ipr = []
    all_aa_lyap = []
    
    for W, data in hh_results.items():
        all_hh_evals.extend(data['evals'])
        all_hh_ipr.extend(data['ipr'])
        all_hh_lyap.extend(data['lyap'])
    
    for W, data in aa_results.items():
        all_aa_evals.extend(data['evals'])
        all_aa_ipr.extend(data['ipr'])
        all_aa_lyap.extend(data['lyap'])
    
    # Create IPR comparison plot
    plt.figure(figsize=(14, 10))
    
    # Calculate common limits for better comparison
    min_energy = min(min(all_hh_evals), min(all_aa_evals))
    max_energy = max(max(all_hh_evals), max(all_aa_evals))
    min_ipr = min(min(all_hh_ipr), min(all_aa_ipr))
    max_ipr = max(max(all_hh_ipr), max(all_aa_ipr))
    
    # Create grids for density estimation
    grid_x = np.linspace(min_energy, max_energy, 200)
    grid_y = np.logspace(np.log10(min_ipr), np.log10(max_ipr), 200)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # Plot IPR density for HH
    plt.subplot(2, 2, 1)
    kde_hh = gaussian_kde([all_hh_evals, np.log10(all_hh_ipr)])
    zz_hh = np.reshape(kde_hh([xx.ravel(), np.log10(yy.ravel())]), xx.shape)
    plt.pcolormesh(xx, yy, zz_hh, shading='auto', cmap='viridis', 
                  norm=mcolors.LogNorm(vmin=1e-3, vmax=zz_hh.max()))
    plt.colorbar(label='Density')
    plt.yscale('log')
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.title('Harper-Hofstadter IPR Density')
    
    # Plot IPR density for AA
    plt.subplot(2, 2, 2)
    kde_aa = gaussian_kde([all_aa_evals, np.log10(all_aa_ipr)])
    zz_aa = np.reshape(kde_aa([xx.ravel(), np.log10(yy.ravel())]), xx.shape)
    plt.pcolormesh(xx, yy, zz_aa, shading='auto', cmap='plasma', 
                  norm=mcolors.LogNorm(vmin=1e-3, vmax=zz_aa.max()))
    plt.colorbar(label='Density')
    plt.yscale('log')
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.title('Aubry-André IPR Density')
    
    # Calculate Lyapunov limits
    min_lyap = min(min(all_hh_lyap), min(all_aa_lyap))
    max_lyap = max(max(all_hh_lyap), max(all_aa_lyap))
    
    # Create grids for Lyapunov density
    grid_x = np.linspace(min_energy, max_energy, 200)
    grid_y = np.linspace(min_lyap, max_lyap, 200)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # Plot Lyapunov density for HH
    plt.subplot(2, 2, 3)
    kde_hh_lyap = gaussian_kde([all_hh_evals, all_hh_lyap])
    zz_hh_lyap = np.reshape(kde_hh_lyap([xx.ravel(), yy.ravel()]), xx.shape)
    plt.pcolormesh(xx, yy, zz_hh_lyap, shading='auto', cmap='viridis', 
                  norm=mcolors.LogNorm(vmin=1e-3, vmax=zz_hh_lyap.max()))
    plt.colorbar(label='Density')
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent')
    plt.title('Harper-Hofstadter Lyapunov Density')
    
    # Plot Lyapunov density for AA
    plt.subplot(2, 2, 4)
    kde_aa_lyap = gaussian_kde([all_aa_evals, all_aa_lyap])
    zz_aa_lyap = np.reshape(kde_aa_lyap([xx.ravel(), yy.ravel()]), xx.shape)
    plt.pcolormesh(xx, yy, zz_aa_lyap, shading='auto', cmap='plasma', 
                  norm=mcolors.LogNorm(vmin=1e-3, vmax=zz_aa_lyap.max()))
    plt.colorbar(label='Density')
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent')
    plt.title('Aubry-André Lyapunov Density')
    
    plt.tight_layout()
    density_path = os.path.join(base_dir, "comparisons", 'density_comparison.png')
    plt.savefig(density_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overlaid scatter comparison
    plt.figure(figsize=(14, 10))
    
    # IPR comparison
    plt.subplot(2, 1, 1)
    plt.scatter(all_hh_evals, all_hh_ipr, s=10, alpha=0.2, 
               label='Harper-Hofstadter', color='blue')
    plt.scatter(all_aa_evals, all_aa_ipr, s=10, alpha=0.2, 
               label='Aubry-André', color='red')
    plt.xlabel('Energy E')
    plt.ylabel('IPR')
    plt.yscale('log')
    plt.title('Combined IPR Comparison (All W Values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lyapunov comparison
    plt.subplot(2, 1, 2)
    plt.scatter(all_hh_evals, all_hh_lyap, s=10, alpha=0.2, 
               label='Harper-Hofstadter', color='blue')
    plt.scatter(all_aa_evals, all_aa_lyap, s=10, alpha=0.2, 
               label='Aubry-André', color='red')
    plt.xlabel('Energy E')
    plt.ylabel('Lyapunov Exponent')
    plt.title('Combined Lyapunov Exponent Comparison (All W Values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(base_dir, "comparisons", 'scatter_comparison.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

# ===================================================================
# Main Simulation
# ===================================================================

def run_comparison_simulation():
    """Run comparison between Harper-Hofstadter and Aubry-André models for various W."""
    # Model parameters
    Nx, Ny = 20, 20  # Harper-Hofstadter dimensions
    total_sites = Nx * Ny
    alpha = 1/5       # Flux per plaquette
    beta = alpha      # For Aubry-André
    lambda_ = 2.0     # Critical point for AA model
    W_vals = [0, 0.5, 1, 2, 5, 10]  # Disorder strengths
    
    # Create directory structure
    base_dir = create_directories(W_vals)
    
    # Generate Hofstadter butterfly
    plot_hofstadter_butterfly(base_dir)
    
    # Results storage
    hh_results = {}
    aa_results = {}
    connection_results = []
    
    for W in W_vals:
        print("="*70)
        print(f"Running simulations for W = {W}")
        print("="*70)
        
        # ===========================================================
        # Harper-Hofstadter simulation
        # ===========================================================
        H_hh, disorder_hh = harper_hofstadter_hamiltonian(Nx, Ny, alpha, W=W)
        evals_hh, evecs_hh = eigh(H_hh)
        ipr_hh = compute_ipr(evecs_hh)
        le_hh = compute_lyapunov(evecs_hh, W)
        
        # Store results for combined comparison
        hh_results[W] = {
            'evals': evals_hh,
            'ipr': ipr_hh,
            'lyap': le_hh
        }
        
        # Plot Harper-Hofstadter specific analyses
        plot_disorder_profile(disorder_hh, W, base_dir)
        plot_dos(evals_hh, W, base_dir)
        plot_level_spacing(evals_hh, W, base_dir)
        
        # Wavefunction visualization (3D and metrics)
        wave_path = os.path.join(base_dir, "wavefunctions", f'wavefunction_3d_W{W}.png')
        plot_3d_wavefunction(evecs_hh, evals_hh, W, wave_path)
        plot_wavefunction_analysis(evecs_hh, evals_hh, W, base_dir)
        
        # Time evolution animation
        create_time_evolution_animation(H_hh, total_sites, W, base_dir, duration=5)
        
        # ===========================================================
        # Aubry-André simulation
        # ===========================================================
        H_aa, disorder_aa = aubry_andre_hamiltonian(total_sites, lambda_, beta, W=W)
        evals_aa, evecs_aa = eigh(H_aa)
        ipr_aa = compute_ipr(evecs_aa)
        le_aa = compute_lyapunov(evecs_aa, W)
        
        # Store results for combined comparison
        aa_results[W] = {
            'evals': evals_aa,
            'ipr': ipr_aa,
            'lyap': le_aa
        }
        
        # Store results for connection demonstration
        connection_results.append({
            'W': W,
            'hh_mean_ipr': np.mean(ipr_hh),
            'aa_mean_ipr': np.mean(ipr_aa),
            'hh_mean_lyap': np.mean(le_hh),
            'aa_mean_lyap': np.mean(le_aa)
        })
        
        print(f"Completed W = {W}\n")
    
    # Create comprehensive comparison plots
    plot_combined_comparisons(hh_results, aa_results, base_dir)
    
    # Plot connection between models
    plot_model_connection(connection_results, base_dir)
    
    print("All simulations completed. Output saved to:", base_dir)

def plot_model_connection(results, base_dir):
    """Plot demonstrating connection between models."""
    W_vals = [r['W'] for r in results]
    hh_ipr = [r['hh_mean_ipr'] for r in results]
    aa_ipr = [r['aa_mean_ipr'] for r in results]
    hh_lyap = [r['hh_mean_lyap'] for r in results]
    aa_lyap = [r['aa_mean_lyap'] for r in results]
    
    plt.figure(figsize=(14, 10))
    
    # IPR comparison
    plt.subplot(2, 1, 1)
    plt.plot(W_vals, hh_ipr, 'bo-', linewidth=2, markersize=8, label='Harper-Hofstadter')
    plt.plot(W_vals, aa_ipr, 'rs--', linewidth=2, markersize=8, label='Aubry-André')
    plt.xlabel('Disorder Strength (W)')
    plt.ylabel('Mean IPR')
    plt.title('Model Connection: IPR Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lyapunov exponent comparison
    plt.subplot(2, 1, 2)
    plt.plot(W_vals, hh_lyap, 'bo-', linewidth=2, markersize=8, label='Harper-Hofstadter')
    plt.plot(W_vals, aa_lyap, 'rs--', linewidth=2, markersize=8, label='Aubry-André')
    plt.xlabel('Disorder Strength (W)')
    plt.ylabel('Mean Lyapunov Exponent')
    plt.title('Model Connection: Lyapunov Exponent Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    conn_path = os.path.join(base_dir, "connection", 'model_connection.png')
    plt.savefig(conn_path, dpi=300, bbox_inches='tight')
    plt.close()

# ===================================================================
# Execution
# ===================================================================

if __name__ == "__main__":
    run_comparison_simulation()