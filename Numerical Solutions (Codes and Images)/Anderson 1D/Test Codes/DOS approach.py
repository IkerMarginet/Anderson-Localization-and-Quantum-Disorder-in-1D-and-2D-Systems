import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde

# Set consistent plotting parameters
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})
COLORS = plt.cm.viridis(np.linspace(0, 1, 6))  # 6 colors for 6 W values

def anderson_hamiltonian(L, W):
    """Create the 1D Anderson model Hamiltonian"""
    diagonal = np.random.uniform(-W/2, W/2, L)
    hopping = np.ones(L-1)
    H = np.diag(diagonal) + np.diag(hopping, k=1) + np.diag(hopping, k=-1)
    return H

def lorentzian_dos(energies, eigenvalues, eta=0.05):
    dos = np.zeros_like(energies)
    for E in eigenvalues:
        dos += eta / (np.pi * ((energies - E)**2 + eta**2))
    return dos / len(eigenvalues)

def kde_dos(energies, eigenvalues, bw_method=0.1):
    kde = gaussian_kde(eigenvalues, bw_method=bw_method)
    return kde(energies)

def histogram_dos(eigenvalues, bins=200, range=(-3, 3)):
    hist, bin_edges = np.histogram(eigenvalues, bins=bins, range=range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    return bin_centers, hist

def calculate_all_methods(W_values, L, num_realizations):
    results = {}
    for i, W in enumerate(W_values):
        print(f"\nCalculating for W = {W}...")
        all_eigenvalues = []
        
        for _ in tqdm(range(num_realizations)):
            H = anderson_hamiltonian(L, W)
            all_eigenvalues.extend(np.linalg.eigvalsh(H))
        
        energies = np.linspace(-3, 3, 1000)
        results[W] = {
            'energies': energies,
            'color': COLORS[i],
            'lorentzian': lorentzian_dos(energies, all_eigenvalues),
            'kde': kde_dos(energies, all_eigenvalues),
            'histogram': histogram_dos(all_eigenvalues)
        }
    return results

def plot_individual_W(results, W, ymax):
    """Plot for a single W showing all methods"""
    data = results[W]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all methods
    ax.plot(data['energies'], data['lorentzian'], color=data['color'], 
            label='Lorentzian', lw=2)
    ax.plot(data['energies'], data['kde'], color=data['color'], 
            label='KDE', linestyle=':', lw=2)
    
    bin_centers, counts = data['histogram']
    ax.bar(bin_centers, counts, width=0.8*(bin_centers[1]-bin_centers[0]),
           color=data['color'], alpha=0.3, label='Histogram')
    
    ax.set_title(f'DOS Comparison for W = {W}', fontsize=14)
    ax.set_xlabel('Energy (E)')
    ax.set_ylabel('DOS')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    plt.savefig(f'dos_methods_W{W}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_W_together(results, W_values, ymax):
    """Plot all W values together (Lorentzian method)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for W in W_values:
        data = results[W]
        ax.plot(data['energies'], data['lorentzian'], 
                color=data['color'], label=f'W = {W}', lw=2)
    
    ax.set_title('DOS for Different Disorder Strengths (Lorentzian Method)', fontsize=14)
    ax.set_xlabel('Energy (E)')
    ax.set_ylabel('DOS')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    plt.savefig('dos_all_W_together.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_methods_comparison_grid(results, W_values, ymax):
    """2x3 grid showing methods comparison for each W"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    for i, W in enumerate(W_values):
        data = results[W]
        ax = axes[i]
        
        ax.plot(data['energies'], data['lorentzian'], color=data['color'], 
                label='Lorentzian', lw=2)
        ax.plot(data['energies'], data['kde'], color=data['color'], 
                label='KDE', linestyle=':', lw=2)
        
        bin_centers, counts = data['histogram']
        ax.bar(bin_centers, counts, width=0.8*(bin_centers[1]-bin_centers[0]),
               color=data['color'], alpha=0.3, label='Histogram')
        
        ax.set_title(f'W = {W}', fontsize=14)
        ax.set_xlabel('Energy (E)')
        if i in [0, 3]:
            ax.set_ylabel('DOS')
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('dos_methods_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Parameters
    L = 1000
    num_realizations = 30
    W_values = [0, 0.5, 1, 2, 5, 10]
    
    # Calculate results
    results = calculate_all_methods(W_values, L, num_realizations)
    
    # Find global ymax for consistent scaling
    ymax = max(np.max(data['lorentzian']) for data in results.values()) * 1.1
    
    # Generate plots
    plot_all_W_together(results, W_values, ymax)          # All W together
    plot_methods_comparison_grid(results, W_values, ymax) # Methods comparison grid
    
    # Individual plots for each W
    for W in W_values:
        plot_individual_W(results, W, ymax)

if __name__ == "__main__":
    main()