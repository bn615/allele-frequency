import numpy as np
import math
import matplotlib.pyplot as plt


# EM algorithm for ABO allele frequency estimation

def em_abo(n1, n2, n3, n4, max_iter=1000, tol=1e-6):
    # n1 = type A count, n2 = type B count, n3 = type AB count, n4 = type O count
    # max_iter = maximum number of iterations
    # tol = tolerance for convergence
    
    N = n1 + n2 + n3 + n4 # total individuals

    # initialize allele frequencies
    # p = freq(A), q = freq(B), r = freq(O)
    p, q, r = initialize_freqs()

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
        p1 = (2 * n11 + n12 + n3) / (2 * N)
        q1 = (2 * n21 + n22 + n3) / (2 * N)
        r1 = (n12 + n22 + 2 * n4) / (2 * N)

        iterations.append((i + 1, p1, q1, r1))

        # Check for convergence
        if (abs(p - p1) + abs(q - q1) + abs(r - r1)) < tol:
            print(f"Converged in {i + 1} iterations.")
            break
        
        p, q, r = p1, q1, r1
    

    return p1, q1, r1, iterations

# def newton_raphson_abo(n1, n2, n3, n4, max_iter=1000, tol=1e-4):
#     # n1 = type A count, n2 = type B count, n3 = type AB count, n4 = type O count
#     # max_iter = maximum number of iterations
#     # tol = tolerance for convergence
    
#     N = n1 + n2 + n3 + n4 # total individuals

#     # initialize allele frequencies
#     # p = freq(A), q = freq(B), r = freq(O)
#     p, q, r = initialize_freqs()
#     lam = 2 * N # Lagrange multiplier for constraint p + q + r = 1

#     iterations = []

#     for i in range(max_iter):
#         # Calculate gradients from lagrange
#         dp = n1 * (2*p + 2*r) / (p**2 + 2*p*r) + n3 / p - lam
#         dq = n2 * (2*q + 2*r) / (q**2 + 2*q*r) + n3 / q - lam
#         dr = 2 * n1 / (p + 2*r) + 2 * n2 / (q + 2*r) + 2 * n4 / (r) - lam
#         constraint = p + q + r - 1
        
#         gradient = np.array([dp, dq, dr, constraint])
        
#         # Jacobian matrix
#         denom1 = (p**2 + 2*p*r)**2
#         denom2 = (q**2 + 2*q*r)**2

#         j11 = n1 * (2 * (p*p + 2*p*r) - 4 * (p + r)**2) / denom1 - n3 / p**2
#         j13 = -2 * n1 / (p + 2*r)**2
#         j22 = n2 * (2 * (q*q + 2*q*r) - 4 * (q + r)**2) / denom2 - n3 / q**2
#         j23 = -2 * n2 / (q + 2*r)**2
#         j31 = -2 * n1 / (p + 2*r)**2
#         j32 = -2 * n2 / (q + 2*r)**2
#         j33 = -4 * n1 / (p + 2*r)**2 - 4 * n2 / (q + 2*r)**2 - 2 * n4 / r**2

#         jacobian = np.array([
#             [j11, 0, j13, -1],
#             [0, j22, j23, -1],
#             [j31, j32, j33, -1],
#             [1, 1, 1, 0]
#         ])

#         # Newton-Raphson step
#         delta = np.linalg.solve(jacobian, -gradient)
#         p += delta[0]
#         q += delta[1]
#         r += delta[2]
#         lam += delta[3]

#         iterations.append((i + 1, p, q, r))

#         if np.linalg.norm(delta) < tol:
#             print(f"Converged in {i + 1} iterations.")
#             break

#     return p, q, r, iterations
        
def newton_raphson_abo(n1, n2, n3, n4, max_iter=100000, tol=1e-4):
    
    N = n1 + n2 + n3 + n4  # total individuals

    # Initialize allele frequencies  
    p, q, r = initialize_freqs()
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
        
        # CORRECTED Jacobian matrix
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
        
        # Update variables
        p += delta[0]
        q += delta[1] 
        r += delta[2]
        lam += delta[3]

        # Print progress
        if i < 15 or i % 10 == 0:
            print(f"{i+1}\t{p:.6f}\t{q:.6f}\t{r:.6f}\t{lam:.1f}\t\t{F_norm:.2e}\t{step_norm:.2e}")

        iterations.append((i + 1, p, q, r, lam, F_norm))

        # Secondary convergence check on step size
        if step_norm < tol:
            print(f"Converged (step size) in {i + 1} iterations.")
            break
            
    else:
        print(f"Warning: Maximum iterations reached without convergence.")

def initialize_freqs():
    # Initial guess for allele frequencies: all equal, could be something different
    return np.array([1/3, 1/3, 1/3])


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
    

def print_results(p, q, r):
    print(f"Estimated allele frequencies:")
    print(f"Frequency of A (p): {p:.5f}")
    print(f"Frequency of B (q): {q:.5f}")
    print(f"Frequency of O (r): {r:.5f}")

if __name__ == "__main__":
    n1 =  186  # type A count
    n2 =  38   # type B count
    n3 =  36   # type AB count
    n4 =  284  # type O count

    print(f"Input data: A = {n1}, B = {n2}, AB = {n3}, O = {n4}")
    print(f"Total sample size: {n1 + n2 + n3 + n4}")

    # p, q, r, iterations = em_abo(n1, n2, n3, n4)
    p, q, r, iterations = newton_raphson_abo(n1, n2, n3, n4)

    print_results(p, q, r)
    plot_convergence(iterations)

