import numpy as np
import math
import matplotlib.pyplot as plt


# EM algorithm for ABO allele frequency estimation

def em_abo(n1, n2, n3, n4, p_init, q_init, r_init, max_iter=1000, tol=1e-6):
    # n1 = type A count, n2 = type B count, n3 = type AB count, n4 = type O count
    # max_iter = maximum number of iterations
    # tol = tolerance for convergence
    
    N = n1 + n2 + n3 + n4 # total individuals

    # initialize allele frequencies
    # p = freq(A), q = freq(B), r = freq(O)
    p, q, r = p_init, q_init, r_init
    iterations = []

    for i in range(max_iter):
        # E step: Calculate expected counts of alleles
        # n11 = AA genotype contribution to A
        n11 = n1 * p / (p + 2 * r)
        # n12 = AO genotype contribution to A 
        n12 = n1 - n11
        # n21 = BB genotype contribution to B
        n21 = n2 * q / (q + 2 * r)
        # n22 = BO genotype contribution to B
        n22 = n2 - n21
        
        # M step: Update allele frequencies
        p_new = (2 * n11 + n12 + n3) / (2 * N)
        q_new = (2 * n21 + n22 + n3) / (2 * N)
        r_new = (n12 + n22 + 2 * n4) / (2 * N)

        # log likelihood calculation
        ll = log_likelihood(n1, n2, n3, n4, p_new, q_new, r_new)

        iterations.append((i + 1, p_new, q_new, r_new, ll))

        # Check for convergence
        if (abs(p - p_new) + abs(q - q_new) + abs(r - r_new)) < tol:
            # print(f"Converged in {i + 1} iterations.")
            break
        
        p, q, r = p_new, q_new, r_new
        

    return p, q, r, ll, iterations


def log_likelihood(n1, n2, n3, n4, p, q, r):
    ll = (n1 * np.log(p**2 + 2*p*r) + 
          n2 * np.log(q**2 + 2*q*r) + 
          n3 * np.log(2*p*q) + 
          n4 * np.log(r**2))
    return ll

def em_multiple_starts(n1, n2, n3, n4, starts=10, max_iter=1000, tol=1e-6):

    best_likelihood = -np.inf
    best_params = None
    best_iterations = None
    results = []
    
    for i in range(starts):
        if i == 0:
            p0, q0, r0 = initialize_freqs(n1, n2, n3, n4)
            method = "standard"
        
        else:
            px, qx, rx = np.random.dirichlet(np.ones(3))  # Random initialization
            p0, q0, r0 = px, qx, rx
            method = "random"

        p, q, r, ll, iterations = em_abo(n1, n2, n3, n4, p0, q0, r0, max_iter, tol)

        results.append({
            'start': i+1,
            'method': method,
            'p': p,
            'q': q,
            'r': r,
            'log_likelihood': ll
        })
        
        # Track the best result
        if ll > best_likelihood:
            best_likelihood = ll
            best_params = (p, q, r)
            best_iterations = iterations
    
    tolerance = 0.01
    converged_to_best = sum(1 for r in results if abs(r["log_likelihood"] - best_likelihood) < tolerance)
    print(f"Best log-likelihood: {best_likelihood:.2f}")
    print(f"Best parameters: p={best_params[0]:.5f}, q={best_params[1]:.5f}, r={best_params[2]:.5f}")
    print(f"Converged to best solution in {converged_to_best} out of {starts} runs.")

    unique_liks = []
    for r in results:
        lik = r['log_likelihood']
        if not any(abs(lik - u) < tolerance for u in unique_liks):
            unique_liks.append(lik)
    
    if len(unique_liks) == 1:
        print("All starts converged to the same solution")
    else:
        print("Distinct log-likelihoods found:")
        for j, lik in enumerate(sorted(unique_liks, reverse=True), 1):
            count = sum(1 for r in results if abs(r['log_likelihood'] - lik) < tolerance)
            print(f"  Solution {j}: {lik:.6f} (found by {count} starts)")
    
    return best_params[0], best_params[1], best_params[2], best_likelihood, best_iterations
        
def newton_raphson_abo(n1, n2, n3, n4, max_iter=100000, tol=1e-4):
    
    N = n1 + n2 + n3 + n4  # total individuals

    # Initialize allele frequencies  
    p, q, r = initialize_freqs(n1, n2, n3, n4)
    lam = 2 * N  # Good lambda initialization
    
    iterations = []
    print(f"Debugged Newton-Raphson: n1={n1}, n2={n2}, n3={n3}, n4={n4}")

    for i in range(max_iter):
        # Bounds checking for numerical stability
        p = max(1e-8, min(0.999, p))
        q = max(1e-8, min(0.999, q))
        r = max(1e-8, min(0.999, r))
        
        # Renormalize if constraint badly violated
        total = p + q + r
        if abs(total - 1.0) > 0.01:
            p, q, r = p/total, q/total, r/total
        
        # Function values F(x) = 0
        F1 = n1 * (2*p + 2*r) / (p**2 + 2*p*r) + n3 / p - lam
        F2 = n2 * (2*q + 2*r) / (q**2 + 2*q*r) + n3 / q - lam
        F3 = 2 * n1 / (p + 2*r) + 2 * n2 / (q + 2*r) + 2 * n4 / r - lam
        F4 = p + q + r - 1
        
        F = np.array([F1, F2, F3, F4])
        F_norm = np.linalg.norm(F)
        
        # Check convergence
        if F_norm < tol:
            print(f"Converged in {i + 1} iterations (||F|| = {F_norm:.2e})")
            break
        
        # Jacobian matrix
        denom1 = (p**2 + 2*p*r)**2
        denom2 = (q**2 + 2*q*r)**2

        # Row 1: ∂F1/∂p, ∂F1/∂q, ∂F1/∂r, ∂F1/∂λ
        j11 = n1 * (2 * (p**2 + 2*p*r) - (2*p + 2*r)**2) / denom1 - n3 / p**2
        j12 = 0
        # BUG FIX: This was the main error!
        j13 = -2 * n1 * p**2 / denom1  # NOT -2*n1/(p+2r)²
        j14 = -1

        # Row 2: ∂F2/∂p, ∂F2/∂q, ∂F2/∂r, ∂F2/∂λ
        j21 = 0
        j22 = n2 * (2 * (q**2 + 2*q*r) - (2*q + 2*r)**2) / denom2 - n3 / q**2
        # BUG FIX: Same issue here
        j23 = -2 * n2 * q**2 / denom2  # NOT -2*n2/(q+2r)²
        j24 = -1

        # Row 3: ∂F3/∂p, ∂F3/∂q, ∂F3/∂r, ∂F3/∂λ (these were correct)
        j31 = -2 * n1 / (p + 2*r)**2
        j32 = -2 * n2 / (q + 2*r)**2
        j33 = -4 * n1 / (p + 2*r)**2 - 4 * n2 / (q + 2*r)**2 - 2 * n4 / r**2
        j34 = -1

        # Row 4: Constraint (correct)
        j41, j42, j43, j44 = 1, 1, 1, 0

        jacobian = np.array([
            [j11, j12, j13, j14],
            [j21, j22, j23, j24],
            [j31, j32, j33, j34],
            [j41, j42, j43, j44]
        ])

        # Solve Newton system with error handling
        try:
            cond_num = np.linalg.cond(jacobian)
            if cond_num > 1e12:
                print(f"Warning: Ill-conditioned Jacobian (cond={cond_num:.2e})")
            
            delta = np.linalg.solve(jacobian, -F)
            
        except np.linalg.LinAlgError:
            print(f"Singular Jacobian at iteration {i+1}")
            # Fallback to gradient descent
            alpha = 0.001
            delta = -alpha * F
        
        # Adaptive step size control
        step_norm = np.linalg.norm(delta)
        if step_norm > 0.1:  # Limit step size
            delta = delta * 0.1 / step_norm
            step_norm = 0.1
        
        # log likelihood for monitoring
        ll = log_likelihood(n1, n2, n3, n4, p, q, r)

        # Update variables
        p += delta[0]
        q += delta[1] 
        r += delta[2]
        lam += delta[3]

        # Print progress
        if i < 15 or i % 10 == 0:
            print(f"{i+1}\t{p:.6f}\t{q:.6f}\t{r:.6f}\t{lam:.1f}\t\t{F_norm:.2e}\t{step_norm:.2e}")

        iterations.append((i+ 1, p, q, r, ll))

        # Secondary convergence check on step size
        if step_norm < tol:
            print(f"Converged (step size) in {i + 1} iterations.")
            break
            
    else:
        print(f"Warning: Maximum iterations reached without convergence.")
    
    return p, q, r, iterations

def initialize_freqs(n1=0, n2=0, n3=0, n4=0):
    # Initial guess for allele frequencies: all equal, could be something different
    total = n1 + n2 + n3 + n4

    r = math.sqrt(n4 / total)

    q = -r + math.sqrt(r**2 + n2 / total)
    p = 1 - q - r

    return np.array([p, q, r])
    # return np.array([1/3, 1/3, 1/3])


def plot_convergence(iterations):
    iterations = np.array(iterations)
    plt.plot(iterations[:, 0], iterations[:, 1], label='p (A)', marker='o')
    plt.plot(iterations[:, 0], iterations[:, 2], label='q (B)', marker='o')
    plt.plot(iterations[:, 0], iterations[:, 3], label='r (O)', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Allele Frequency')
    plt.title('Convergence of Allele Frequencies')
    plt.legend()
    plt.grid()
    plt.show()
    

def print_results(p, q, r, ll = None):
    print(f"Estimated allele frequencies:")
    print(f"Frequency of A (p): {p:.5f}")
    print(f"Frequency of B (q): {q:.5f}")
    print(f"Frequency of O (r): {r:.5f}")
    if ll is not None:
        print(f"  Log-likelihood: {ll:.4f}")

if __name__ == "__main__":
    # n1 =  2
    # n2 =  1
    # n3 =  0
    # n4 =  5000
    n1 =  186  # type A count
    n2 =  38   # type B count
    n3 =  36   # type AB count
    n4 =  284  # type O count

    print(f"Input data: A = {n1}, B = {n2}, AB = {n3}, O = {n4}")
    print(f"Total sample size: {n1 + n2 + n3 + n4}")


    # p_init, q_init, r_init = initialize_freqs(n1, n2, n3, n4)

    em_starts = 10
    p, q, r, ll, iterations = em_multiple_starts(n1, n2, n3, n4, starts=em_starts)
    # p, q, r, ll, iterations = em_abo(n1, n2, n3, n4, p_init, q_init, r_init)
    # p, q, r, iterations = newton_raphson_abo(n1, n2, n3, n4)

    print_results(p, q, r, ll)
    # plot_convergence(iterations)

