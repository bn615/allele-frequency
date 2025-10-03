import numpy as np
import math
import matplotlib.pyplot as plt

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

def bootstrap_em(n1, n2, n3, n4, n_boot=1000, c_level=0.95):
    
    n = n1 + n2 + n3 + n4

    individuals = np.array(['A'] * n1 + ['B'] * n2 + ['AB'] * n3 + ['O'] * n4)

    p_hat, q_hat, r_hat = em_abo_simple(n1, n2, n3, n4)

    boot_p = []
    boot_q = []
    boot_r = []
    for i in range(n_boot):
        sample = np.random.choice(individuals, size=n, replace=True)
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

    results = ci_interval(p_hat, q_hat, r_hat, boot_p, boot_q, boot_r, c_level, n)
    print_results(results)
    return results


def ci_interval(p_hat, q_hat, r_hat, boot_p, boot_q, boot_r, c_level, n):
    alpha = 1 - c_level
    results = {}

    # standard error
    se_p = np.std(boot_p, ddof=1)
    se_q = np.std(boot_q, ddof=1)
    se_r = np.std(boot_r, ddof=1)

    # percentile confidence interval
    p_ci = (np.percentile(boot_p, 100 * alpha / 2), np.percentile(boot_p, 100 * (1 - alpha / 2)))
    q_ci = (np.percentile(boot_q, 100 * alpha / 2), np.percentile(boot_q, 100 * (1 - alpha / 2)))
    r_ci = (np.percentile(boot_r, 100 * alpha / 2), np.percentile(boot_r, 100 * (1 - alpha / 2)))

    results["p"] = {"original": p_hat, "ci": p_ci, "se": se_p, "bootstrap": boot_p}
    results["q"] = {"original": q_hat, "ci": q_ci, "se": se_q, "bootstrap": boot_q}
    results["r"] = {"original": r_hat, "ci": r_ci, "se": se_r, "bootstrap": boot_r}

    return results

def print_results(results):
    for allele, res in results.items():
        print(f"Allele {allele}:")
        print(f"  Estimate: {res['original']:.5f}")
        print(f"  Std. Error: {res['se']:.5f}")
        print(f"  {95}% CI: ({res['ci'][0]:.5f}, {res['ci'][1]:.5f})")

def plot_bootstrap(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alleles = ['p', 'q', 'r']
    titles = ['Allele A Frequency (p)', 'Allele B Frequency (q)', 'Allele O Frequency (r)']
    
    for i, allele in enumerate(alleles):
        ax = axes[i]
        boot_values = results[allele]['bootstrap']
        ax.hist(boot_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(results[allele]['original'], color='red', linestyle='dashed', linewidth=2)
        ax.axvline(results[allele]['ci'][0], color='green', linestyle='dashed', linewidth=2)
        ax.axvline(results[allele]['ci'][1], color='green', linestyle='dashed', linewidth=2)
        ax.set_title(titles[i])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n1 =  186  # type A count
    n2 =  38   # type B count
    n3 =  36   # type AB count
    n4 =  284  # type O count

    results = bootstrap_em(n1, n2, n3, n4, n_boot=1000, c_level=0.95)
    plot_bootstrap(results)