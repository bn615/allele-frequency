import time
import numpy as np
import torch
from scipy import stats as scipy_stats
import NN
import bootstrap_EM
import EM_raphson_comparison

def compare_methods_efficiency(n1, n2, n3, n4, n_trials=100, model_path=None, true_values=None):
    """Compare computational efficiency of EM, NR, and NN

    Args:
        n1, n2, n3, n4: Blood type counts
        n_trials: Number of trials to run
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

    # Prepare NN input
    X_test = np.array([[n1, n2, n3, n4]], dtype=np.float32)
    X_test = X_test / X_test.sum()
    
    for trial in range(n_trials):
        # EM timing
        start = time.perf_counter()
        p_em, q_em, r_em = bootstrap_EM.em_abo_simple(n1, n2, n3, n4)
        em_time = time.perf_counter() - start
        results['em']['times'].append(em_time)
        results['em']['estimates'].append([p_em, q_em, r_em])
        
        # Newton-Raphson timing
        start = time.perf_counter()
        p_nr, q_nr, r_nr, _ = EM_raphson_comparison.newton_raphson_abo(n1, n2, n3, n4)
        nr_time = time.perf_counter() - start
        results['nr']['times'].append(nr_time)
        results['nr']['estimates'].append([p_nr, q_nr, r_nr])
        
        # Neural Network timing
        start = time.perf_counter()
        prediction = trainer.predict(X_test)
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
            se = std_estimates / np.sqrt(n_trials)  # Standard error
            t_stat = bias / se
            p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stat), df=n_trials-1))

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
            null_p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(null_t_stat), df=n_trials-1))
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

    # Compare inference times
    # If you don't know true values, omit the true_values parameter or set to None
    comparison_results = compare_methods_efficiency(
        n1=186, n2=38, n3=36, n4=284,
        n_trials=100,
        model_path="abo_model.pth"
        # true_values=(0.2814, 0.1061, 0.6125)  # Uncomment if you know true values
    )