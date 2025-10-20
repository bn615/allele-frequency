import time
import numpy as np
import torch
from scipy import stats as scipy_stats
import NN
import bootstrap_EM
import EM_raphson_comparison

def compare_methods_efficiency(n1, n2, n3, n4, n_bootstrap=1000, model_path=None, true_values=None):
    """Compare computational efficiency of EM, NR, and NN using nonparametric bootstrap

    Args:
        n1, n2, n3, n4: Blood type counts
        n_bootstrap: Number of bootstrap resamples
        model_path: Path to trained NN model
        true_values: Optional tuple (p_true, q_true, r_true) for bias and power calculation
    """

    results = {
        'em': {'times': [], 'estimates': []},
        'nr': {'times': [], 'estimates': []},
        'nn': {'times': [], 'estimates': []}
    }

    # Initialize and load neural network model
    model = NN.ABONN()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    trainer = NN.ABOTrainer(model)

    # Create individual-level data for bootstrap resampling
    n = n1 + n2 + n3 + n4
    individuals = np.array(['A'] * n1 + ['B'] * n2 + ['AB'] * n3 + ['O'] * n4)

    # Perform nonparametric bootstrap
    for trial in range(n_bootstrap):
        # Resample individuals with replacement
        sample = np.random.choice(individuals, size=n, replace=True)
        n1_boot = np.sum(sample == 'A')
        n2_boot = np.sum(sample == 'B')
        n3_boot = np.sum(sample == 'AB')
        n4_boot = np.sum(sample == 'O')

        # EM timing
        start = time.perf_counter()
        p_em, q_em, r_em = bootstrap_EM.em_abo_simple(n1_boot, n2_boot, n3_boot, n4_boot)
        em_time = time.perf_counter() - start
        results['em']['times'].append(em_time)
        results['em']['estimates'].append([p_em, q_em, r_em])

        # Newton-Raphson timing
        start = time.perf_counter()
        p_nr, q_nr, r_nr, _ = EM_raphson_comparison.newton_raphson_abo(n1_boot, n2_boot, n3_boot, n4_boot, verbose=False)
        nr_time = time.perf_counter() - start
        results['nr']['times'].append(nr_time)
        results['nr']['estimates'].append([p_nr, q_nr, r_nr])

        # Neural Network timing
        X_boot = np.array([[n1_boot, n2_boot, n3_boot, n4_boot]], dtype=np.float32)
        X_boot = X_boot / X_boot.sum()
        start = time.perf_counter()
        prediction = trainer.predict(X_boot)
        nn_time = time.perf_counter() - start
        results['nn']['times'].append(nn_time)
        results['nn']['estimates'].append(prediction[0].tolist())

    # Calculate summary statistics for each method
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    if true_values is not None:
        p_true, q_true, r_true = true_values
        print(f"True values: p={p_true:.4f}, q={q_true:.4f}, r={r_true:.4f}\n")

    for method in ['em', 'nr', 'nn']:
        times = np.array(results[method]['times'])
        estimates = np.array(results[method]['estimates'])

        # Calculate statistics for each allele (p, q, r)
        mean_estimates = estimates.mean(axis=0)
        std_estimates = estimates.std(axis=0)
        variance = std_estimates ** 2

        # Calculate 95% confidence intervals
        ci_lower = np.percentile(estimates, 2.5, axis=0)
        ci_upper = np.percentile(estimates, 97.5, axis=0)

        print(f"\n{method.upper()} Method:")
        print(f"  Computational Efficiency: {times.mean()*1000:.3f} ms")

        print(f"\n  Estimates (mean +/- std):")
        print(f"    p: {mean_estimates[0]:.4f} +/- {std_estimates[0]:.4f}")
        print(f"    q: {mean_estimates[1]:.4f} +/- {std_estimates[1]:.4f}")
        print(f"    r: {mean_estimates[2]:.4f} +/- {std_estimates[2]:.4f}")

        print(f"\n  95% Confidence Intervals:")
        print(f"    p: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
        print(f"    q: [{ci_lower[1]:.4f}, {ci_upper[1]:.4f}]")
        print(f"    r: [{ci_lower[2]:.4f}, {ci_upper[2]:.4f}]")

        print(f"\n  Statistical Efficiency (variance, lower is better):")
        print(f"    p: {variance[0]:.6f}")
        print(f"    q: {variance[1]:.6f}")
        print(f"    r: {variance[2]:.6f}")

        # Calculate bias and power if true values are provided
        if true_values is not None:
            bias = mean_estimates - np.array([p_true, q_true, r_true])
            print(f"\n  Bias (estimate - true):")
            print(f"    p: {bias[0]:+.4f}")
            print(f"    q: {bias[1]:+.4f}")
            print(f"    r: {bias[2]:+.4f}")

            # Coverage: whether CI contains true value
            ci_contains_true = np.array([
                (ci_lower[0] <= p_true <= ci_upper[0]),
                (ci_lower[1] <= q_true <= ci_upper[1]),
                (ci_lower[2] <= r_true <= ci_upper[2])
            ])

            print(f"\n  Coverage (CI contains true value):")
            print(f"    p: {'Yes' if ci_contains_true[0] else 'No'}")
            print(f"    q: {'Yes' if ci_contains_true[1] else 'No'}")
            print(f"    r: {'Yes' if ci_contains_true[2] else 'No'}")

            # Hypothesis testing: t-test for H0: estimate = true_value
            se = std_estimates / np.sqrt(n_bootstrap)  # Standard error
            t_stat = bias / se
            p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stat), df=n_bootstrap-1))

            print(f"\n  Hypothesis Test (H0: estimate = true value):")
            print(f"    p: t={t_stat[0]:+.2f}, p-value={p_values[0]:.4f} {'*' if p_values[0] < 0.05 else ''}")
            print(f"    q: t={t_stat[1]:+.2f}, p-value={p_values[1]:.4f} {'*' if p_values[1] < 0.05 else ''}")
            print(f"    r: t={t_stat[2]:+.2f}, p-value={p_values[2]:.4f} {'*' if p_values[2] < 0.05 else ''}")
            print(f"    (* indicates significant bias at alpha=0.05)")

            # Power calculation: proportion of significant results if there was a true effect
            # For power, we need to detect if estimate differs from null (e.g., differs from 0.33)
            # Here we'll calculate empirical power by checking how often we'd reject H0
            # if the null hypothesis were a different value (e.g., equal frequencies)
            null_value = 1/3  # Null hypothesis: all alleles equal
            null_bias = mean_estimates - null_value
            null_t_stat = null_bias / se
            null_p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(null_t_stat), df=n_bootstrap-1))
            power = (null_p_values < 0.05).astype(float)

            print(f"\n  Power (ability to detect difference from H0: freq=1/3):")
            print(f"    p: {power[0]*100:.1f}% {'(detected)' if power[0] > 0 else '(not detected)'}")
            print(f"    q: {power[1]*100:.1f}% {'(detected)' if power[1] > 0 else '(not detected)'}")
            print(f"    r: {power[2]*100:.1f}% {'(detected)' if power[2] > 0 else '(not detected)'}")

    return results


def train_neural_network_with_timing(n_train=10000, n_val=2000, n_epochs=100, batch_size=64,
                                     distribution="realistic", save_path=None, verbose=True):
    """Train a neural network and measure training time"""

    if verbose:
        print("Generating training data...")
    data_gen_start = time.perf_counter()
    generator = NN.DataGenerator()
    train_loader, val_loader, _ = generator.create_dataloader(
        n_train=n_train,
        n_val=n_val,
        n_test=0,
        batch_size=batch_size,
        sample_size_range=(100, 1000),
        distribution=distribution
    )
    data_gen_time = time.perf_counter() - data_gen_start
    if verbose:
        print(f"Data generation completed in {data_gen_time:.2f} seconds")

    if verbose:
        print("\nTraining neural network...")
    model = NN.ABONN()
    trainer = NN.ABOTrainer(model)

    training_start = time.perf_counter()
    history = trainer.train(train_loader, val_loader, n_epochs=n_epochs)
    training_time = time.perf_counter() - training_start

    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Average time per epoch: {training_time/n_epochs:.3f} seconds")

    if save_path:
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return {
        'model': model,
        'trainer': trainer,
        'history': history,
        'data_gen_time': data_gen_time,
        'training_time': training_time,
        'time_per_epoch': training_time / n_epochs
    }


def simulate_coverage_and_power(n_simulations=100, sample_size=544, n_bootstrap=1000,
                                distribution="realistic", model_path=None, confidence_level=0.95):
    """
    Simulation study to test coverage probability and power using Dirichlet distributions

    Args:
        n_simulations: Number of simulation replicates
        sample_size: Sample size for each simulated dataset
        n_bootstrap: Number of bootstrap resamples per simulation
        distribution: "uniform" (alpha=[1,1,1]) or "realistic" (alpha=[7,2,20])
        model_path: Path to trained NN model
        confidence_level: Confidence level for CIs (default 0.95)

    Returns:
        Dictionary with coverage probabilities and power for each method
    """

    # Initialize data generator
    generator = NN.DataGenerator()

    # Load neural network model if provided
    if model_path:
        model = NN.ABONN()
        model.load_state_dict(torch.load(model_path))
        trainer = NN.ABOTrainer(model)
    else:
        model = None
        trainer = None

    # Storage for results
    coverage_results = {
        'em': {'p': [], 'q': [], 'r': []},
        'nr': {'p': [], 'q': [], 'r': []},
        'nn': {'p': [], 'q': [], 'r': []}
    }

    power_results = {
        'em': {'p': [], 'q': [], 'r': []},
        'nr': {'p': [], 'q': [], 'r': []},
        'nn': {'p': [], 'q': [], 'r': []}
    }

    alpha = (1 - confidence_level) / 2

    print(f"\n{'='*80}")
    print(f"SIMULATION STUDY: Coverage and Power Analysis")
    print(f"{'='*80}")
    print(f"Distribution: {distribution}")
    print(f"Sample size: {sample_size}")
    print(f"Bootstrap resamples: {n_bootstrap}")
    print(f"Simulation replicates: {n_simulations}")
    print(f"Confidence level: {confidence_level*100}%\n")

    for sim in range(n_simulations):
        # Generate true allele frequencies from Dirichlet
        if distribution == "uniform":
            alpha_dirichlet = [1.0, 1.0, 1.0]
        elif distribution == "realistic":
            alpha_dirichlet = [7.0, 2.0, 20.0]
        else:
            raise ValueError("Distribution must be 'uniform' or 'realistic'")

        true_freqs = np.random.dirichlet(alpha_dirichlet)
        p_true, q_true, r_true = true_freqs

        # Generate blood type counts from these frequencies
        n1, n2, n3, n4 = generator.generate_blood_types(p_true, q_true, r_true, sample_size)

        # Get point estimates from the original data (no bootstrap)
        point_estimates = {}
        p_em, q_em, r_em = bootstrap_EM.em_abo_simple(n1, n2, n3, n4)
        point_estimates['em'] = np.array([p_em, q_em, r_em])

        p_nr, q_nr, r_nr, _ = EM_raphson_comparison.newton_raphson_abo(n1, n2, n3, n4, verbose=False)
        point_estimates['nr'] = np.array([p_nr, q_nr, r_nr])

        if trainer:
            X_orig = np.array([[n1, n2, n3, n4]], dtype=np.float32)
            X_orig = X_orig / X_orig.sum()
            prediction = trainer.predict(X_orig)
            point_estimates['nn'] = np.array(prediction[0])

        # Run bootstrap for CI estimation
        results = compare_methods_efficiency_silent(
            n1, n2, n3, n4,
            n_bootstrap=n_bootstrap,
            model_path=model_path if trainer else None,
            trainer=trainer
        )

        # Check coverage and power for each method
        for method in ['em', 'nr', 'nn']:
            if method == 'nn' and not trainer:
                continue

            estimates = np.array(results[method]['estimates'])
            ci_lower = np.percentile(estimates, 100 * alpha, axis=0)
            ci_upper = np.percentile(estimates, 100 * (1 - alpha), axis=0)

            # Coverage: does CI contain true value?
            coverage_results[method]['p'].append(ci_lower[0] <= p_true <= ci_upper[0])
            coverage_results[method]['q'].append(ci_lower[1] <= q_true <= ci_upper[1])
            coverage_results[method]['r'].append(ci_lower[2] <= r_true <= ci_upper[2])

            # Power: can we reject H0: estimate = expected_value (from Dirichlet mean)?
            # Using point estimates (not bootstrap mean) to test against null
            # For Dirichlet(alpha), E[p_i] = alpha_i / sum(alpha)
            if distribution == "realistic":
                # Expected values: [7, 2, 20] / 29
                null_values = np.array([7.0, 2.0, 20.0]) / 29.0
            else:  # uniform
                null_values = np.array([1/3, 1/3, 1/3])

            # Use bootstrap SE to test if point estimate differs from null
            se = estimates.std(axis=0, ddof=1)
            t_stat = (point_estimates[method] - null_values) / se
            # For hypothesis testing, SE should not be divided by sqrt(n_bootstrap)
            # since we're using the bootstrap distribution directly
            p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stat), df=n_bootstrap-1))

            power_results[method]['p'].append(p_values[0] < 0.05)
            power_results[method]['q'].append(p_values[1] < 0.05)
            power_results[method]['r'].append(p_values[2] < 0.05)

        if (sim + 1) % 10 == 0:
            print(f"Completed {sim + 1}/{n_simulations} simulations...")

    # Calculate and report results
    print(f"\n{'='*80}")
    print("COVERAGE PROBABILITY RESULTS (should be ~95%)")
    print(f"{'='*80}\n")

    for method in ['em', 'nr', 'nn']:
        print(f"{method.upper()} Method:")
        for allele in ['p', 'q', 'r']:
            coverage = np.mean(coverage_results[method][allele]) * 100
            print(f"  {allele}: {coverage:.1f}%")
        print()

    # Determine null hypothesis for display
    if distribution == "realistic":
        null_display = "H0: p=0.241, q=0.069, r=0.690 (Dirichlet mean)"
    else:
        null_display = "H0: p=q=r=0.333 (equal frequencies)"

    print(f"{'='*80}")
    print(f"POWER RESULTS (ability to detect difference from {null_display})")
    print(f"{'='*80}\n")

    for method in ['em', 'nr', 'nn']:
        print(f"{method.upper()} Method:")
        for allele in ['p', 'q', 'r']:
            power = np.mean(power_results[method][allele]) * 100
            print(f"  {allele}: {power:.1f}%")
        print()

    return {
        'coverage': coverage_results,
        'power': power_results
    }


def compare_methods_efficiency_silent(n1, n2, n3, n4, n_bootstrap=1000, model_path=None, trainer=None):
    """Silent version of compare_methods_efficiency for simulations (no printing)"""

    results = {
        'em': {'times': [], 'estimates': []},
        'nr': {'times': [], 'estimates': []},
        'nn': {'times': [], 'estimates': []}
    }

    # Initialize and load neural network model
    if trainer is None and model_path:
        model = NN.ABONN()
        model.load_state_dict(torch.load(model_path))
        trainer = NN.ABOTrainer(model)

    # Create individual-level data for bootstrap resampling
    n = n1 + n2 + n3 + n4
    individuals = np.array(['A'] * n1 + ['B'] * n2 + ['AB'] * n3 + ['O'] * n4)

    # Perform nonparametric bootstrap
    for trial in range(n_bootstrap):
        # Resample individuals with replacement
        sample = np.random.choice(individuals, size=n, replace=True)
        n1_boot = np.sum(sample == 'A')
        n2_boot = np.sum(sample == 'B')
        n3_boot = np.sum(sample == 'AB')
        n4_boot = np.sum(sample == 'O')

        # EM timing
        start = time.perf_counter()
        p_em, q_em, r_em = bootstrap_EM.em_abo_simple(n1_boot, n2_boot, n3_boot, n4_boot)
        em_time = time.perf_counter() - start
        results['em']['times'].append(em_time)
        results['em']['estimates'].append([p_em, q_em, r_em])

        # Newton-Raphson timing
        start = time.perf_counter()
        p_nr, q_nr, r_nr, _ = EM_raphson_comparison.newton_raphson_abo(n1_boot, n2_boot, n3_boot, n4_boot, verbose=False)
        nr_time = time.perf_counter() - start
        results['nr']['times'].append(nr_time)
        results['nr']['estimates'].append([p_nr, q_nr, r_nr])

        # Neural Network timing (if available)
        if trainer:
            X_boot = np.array([[n1_boot, n2_boot, n3_boot, n4_boot]], dtype=np.float32)
            X_boot = X_boot / X_boot.sum()
            start = time.perf_counter()
            prediction = trainer.predict(X_boot)
            nn_time = time.perf_counter() - start
            results['nn']['times'].append(nn_time)
            results['nn']['estimates'].append(prediction[0].tolist())

    return results


if __name__ == "__main__":
    # Train model with timing (silent mode)
    training_results = train_neural_network_with_timing(
        n_train=10000,
        n_val=2000,
        n_epochs=100,
        distribution="realistic",
        save_path="abo_model.pth",
        verbose=False  # Suppress training output
    )

    # Compare methods using nonparametric bootstrap on real data
    # For real data, we don't use true_values since they're unknown
    print("\n" + "="*80)
    print("REAL DATA ANALYSIS")
    print("="*80)
    # comparison_results = compare_methods_efficiency(
    #     n1=186, n2=38, n3=36, n4=284,
    #     n_bootstrap=1000,
    #     model_path="abo_model.pth"
    #     # Note: true_values parameter NOT used for real data!
    # )

    # Run simulation study for coverage and power analysis
    # print("\n\n")
    simulation_results = simulate_coverage_and_power(
        n_simulations=50,
        sample_size=544,  # Same as real data (186+38+36+284)
        n_bootstrap=100,
        distribution="realistic",
        model_path="abo_model.pth"
    )