import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from datetime import datetime
import os


def em_abo_simple(n1, n2, n3, n4, max_iter=1000, tol=1e-6):
    N = n1 + n2 + n3 + n4
    
    # Initialize using Method of Moments
    p, q, r = initialize_freqs(n1, n2, n3, n4)
    
    for _ in range(max_iter):
        # E-step
        n11 = n1 * p / (p + 2 * r)
        n21 = n2 * q / (q + 2 * r)
        
        # M-step
        p_new = (2 * n11 + (n1 - n11) + n3) / (2 * N)
        q_new = (2 * n21 + (n2 - n21) + n3) / (2 * N)
        r_new = ((n1 - n11) + (n2 - n21) + 2 * n4) / (2 * N)
        
        # Check convergence
        if abs(p_new - p) + abs(q_new - q) + abs(r_new - r) < tol:
            return p_new, q_new, r_new
        
        p, q, r = p_new, q_new, r_new
    
    return p, q, r


def log_likelihood(n1, n2, n3, n4, p, q, r):
    ll = (n1 * np.log(p**2 + 2*p*r) + 
          n2 * np.log(q**2 + 2*q*r) + 
          n3 * np.log(2*p*q) + 
          n4 * np.log(r**2))
    return ll

def initialize_freqs(n1=0, n2=0, n3=0, n4=0):
    # Initial guess for allele frequencies: all equal, could be something different
    total = n1 + n2 + n3 + n4

    r = math.sqrt(n4 / total)

    q = -r + math.sqrt(r**2 + n2 / total)
    p = 1 - q - r

    return np.array([p, q, r])
    # return np.array([1/3, 1/3, 1/3])

def bootstrap_em(n1, n2, n3, n4, sample_size = 100, n_boot=1000, c_level=0.95, output_dir="bootstrap_results"):

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    n = n1 + n2 + n3 + n4

    individuals = np.array(['A'] * n1 + ['B'] * n2 + ['AB'] * n3 + ['O'] * n4)

    p_hat, q_hat, r_hat = em_abo_simple(n1, n2, n3, n4)

    boot_p = []
    boot_q = []
    boot_r = []
    for i in range(n_boot):
        sample = np.random.choice(individuals, size=sample_size, replace=True)
        n1_boot = np.sum(sample == 'A')
        n2_boot = np.sum(sample == 'B')
        n3_boot = np.sum(sample == 'AB')
        n4_boot = np.sum(sample == 'O')

        p_b, q_b, r_b = em_abo_simple(n1_boot, n2_boot, n3_boot, n4_boot)

        boot_p.append(p_b)
        boot_q.append(q_b)
        boot_r.append(r_b)

    boot_p = np.array(boot_p)
    boot_q = np.array(boot_q)
    boot_r = np.array(boot_r)

    results = compute_statistics(p_hat, q_hat, r_hat, boot_p, boot_q, boot_r, c_level)

    export_plots(results, output_dir, timestamp)
    export_text_report(results, n1, n2, n3, n4, n_boot, c_level, output_dir, timestamp)

    # print_results(results)
    return results


def compute_statistics(p_hat, q_hat, r_hat, boot_p, boot_q, boot_r, c_level):
    alpha = 1 - c_level
    results = {}
    for allele, boot_data, original in [('p', boot_p, p_hat), ('q', boot_q, q_hat), ('r', boot_r, r_hat)]:
            se = np.std(boot_data, ddof = 1)
            bias = np.mean(boot_data) - original
            if original != 0:
                cv = se / original
            else:
                cv = np.inf
            ci = np.percentile(boot_data, [100 * alpha/2, 100 * (1 - alpha/2)])
            
            results[allele] = {
                'original': original,
                'standard_error': se,
                'bias': bias,
                'relative_bias_percent': (bias / original * 100) if original != 0 else 0,
                'coefficient_of_variation': cv,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'ci_width': ci[1] - ci[0],
                'bootstrap_samples': boot_data
            }
    return results


def export_plots(results, output_dir, timestamp):
    """Create and export all plots"""
    
    # 1. Histograms
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    params = ['p', 'q', 'r']
    labels = ['p (A allele)', 'q (B allele)', 'r (O allele)']
    colors = ['blue', 'green', 'red']
    
    for ax, param, label, color in zip(axes, params, labels, colors):
        r = results[param]
        boot_data = r['bootstrap_samples']
        
        ax.hist(boot_data, bins=50, density=True, alpha=0.6, 
               color=color, edgecolor='black')
        ax.axvline(r['original'], color='red', linewidth=2, 
                  linestyle='--', label='Estimate')
        ax.axvspan(r['ci_lower'], r['ci_upper'], alpha=0.2, 
                  color='yellow', label='95% CI')
        
        # Add normal curve
        mu, sigma = np.mean(boot_data), np.std(boot_data)
        x = np.linspace(boot_data.min(), boot_data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2)
        
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Bootstrap Distribution: {label}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histograms_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    box_data = [results['p']['bootstrap_samples'],
                results['q']['bootstrap_samples'],
                results['r']['bootstrap_samples']]
    
    bp = ax.boxplot(box_data, labels=['p (A)', 'q (B)', 'r (O)'],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', 
                                  markersize=8, label='Mean'))
    
    # Color boxes
    colors_box = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    # Add original estimates as diamonds
    for i, param in enumerate(['p', 'q', 'r'], 1):
        ax.plot(i, results[param]['original'], 'rD', markersize=10, 
               label='Original Estimate' if i == 1 else '')
    
    ax.set_ylabel('Allele Frequency', fontsize=12)
    ax.set_title('Bootstrap Distribution Box Plots', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplots_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined plot (histograms + boxplots)
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (param, label, color) in enumerate(zip(params, labels, colors)):
        r = results[param]
        boot_data = r['bootstrap_samples']
        
        # Top row: Histograms
        ax1 = axes[0, i]
        ax1.hist(boot_data, bins=50, density=True, alpha=0.6, 
                color=color, edgecolor='black')
        ax1.axvline(r['original'], color='red', linewidth=2, linestyle='--')
        ax1.axvspan(r['ci_lower'], r['ci_upper'], alpha=0.2, color='yellow')
        ax1.set_xlabel(label, fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'Distribution: {label}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Bottom row: Box plots
        ax2 = axes[1, i]
        bp = ax2.boxplot([boot_data], patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        ax2.axhline(r['original'], color='red', linewidth=2, linestyle='--', 
                   label='Estimate')
        ax2.axhspan(r['ci_lower'], r['ci_upper'], alpha=0.2, color='yellow',
                   label='95% CI')
        ax2.set_ylabel(label, fontsize=11)
        ax2.set_title(f'Box Plot: {label}', fontsize=12)
        ax2.set_xticklabels([''])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comprehensive Bootstrap Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_plots_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def export_text_report(results, n1, n2, n3, n4, n_bootstrap, 
                      confidence_level, output_dir, timestamp):
    """Export comprehensive text report"""
    
    filepath = os.path.join(output_dir, f'report_{timestamp}.txt')
    
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BOOTSTRAP ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("INPUT DATA:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Type A (n1):  {n1}\n")
        f.write(f"Type B (n2):  {n2}\n")
        f.write(f"Type AB (n3): {n3}\n")
        f.write(f"Type O (n4):  {n4}\n")
        f.write(f"Total (n):    {n1+n2+n3+n4}\n\n")
        
        f.write("BOOTSTRAP PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of replicates: {n_bootstrap}\n")
        f.write(f"Confidence level:     {confidence_level*100}%\n")
        f.write(f"Method:               Non-parametric (resampling)\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for param_name, allele_name in [('p', 'A'), ('q', 'B'), ('r', 'O')]:
            r = results[param_name]
            
            f.write(f"{allele_name} ALLELE FREQUENCY ({param_name}):\n")
            f.write("-" * 80 + "\n")
            f.write(f"Point estimate:        {r['original']:.6f}\n")
            f.write(f"Standard error:        {r['standard_error']:.6f}\n")
            f.write(f"Bias:                  {r['bias']:.6f}\n")
            f.write(f"Relative bias:         {r['relative_bias_percent']:.2f}%\n")
            f.write(f"Coefficient of var:    {r['coefficient_of_variation']:.4f}\n")
            f.write(f"95% CI lower:          {r['ci_lower']:.6f}\n")
            f.write(f"95% CI upper:          {r['ci_upper']:.6f}\n")
            f.write(f"CI width:              {r['ci_width']:.6f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("FOR PUBLICATION:\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Allele frequency estimates (n={n1+n2+n3+n4}) with {confidence_level*100}% ")
        f.write(f"confidence intervals\n")
        f.write(f"from {n_bootstrap} bootstrap replicates:\n\n")
        
        for param_name, allele_name in [('p', 'A'), ('q', 'B'), ('r', 'O')]:
            r = results[param_name]
            f.write(f"{allele_name} allele: {r['original']:.4f} ")
            f.write(f"(95% CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], ")
            f.write(f"SE = {r['standard_error']:.4f})\n")

# def plot_bootstrap(results):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     alleles = ['p', 'q', 'r']
#     titles = ['Allele A Frequency (p)', 'Allele B Frequency (q)', 'Allele O Frequency (r)']
    
#     for i, allele in enumerate(alleles):
#         ax = axes[i]
#         boot_values = results[allele]['bootstrap']
#         ax.hist(boot_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
#         ax.axvline(results[allele]['original'], color='red', linestyle='dashed', linewidth=2)
#         ax.axvline(results[allele]['ci'][0], color='green', linestyle='dashed', linewidth=2)
#         ax.axvline(results[allele]['ci'][1], color='green', linestyle='dashed', linewidth=2)
#         ax.set_title(titles[i])
#         ax.set_xlabel('Frequency')
#         ax.set_ylabel('Count')
    
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    n1 =  186  # type A count
    n2 =  38   # type B count
    n3 =  36   # type AB count
    n4 =  284  # type O count

    results = bootstrap_em(n1, n2, n3, n4, sample_size= 100, n_boot=1000, c_level=0.95, output_dir="bootstrap_results")
    # plot_bootstrap(results)