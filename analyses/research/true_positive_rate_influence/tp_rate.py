"""Test for examining the influence of the true positive rate on the reliability of the identification
"""
from idsys import IdMetrics, create_id_system, generate_test_messages
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - TRUE POSITIVE RATE INFLUENCE")
    print("=" * 50)

    # Create systems as a dictionary for compare_systems
    systems = {
        "RSID": create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2}),
        "RMID": create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2}),
        "SHA1ID": create_id_system("SHA1ID", {"gf_exp": 8})
    }

    # Generate test messages
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=10000)

    p_true_positives = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # True positive rates to evaluate
    
    # Store results for each system
    system_results = {name: {'tp_rates': [], 'reliability': [], 'fpr': []} for name in systems.keys()}

    num_trials = 1000000

    for tp_rate in p_true_positives:
        print(f"\nEvaluating with True Positive Rate: {tp_rate}")

        # Evaluate all systems with current true positive rate
        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            num_trials=num_trials,
            timing_iterations=0,
            p_true_positive=tp_rate
        )

        # Store results for each system
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['tp_rates'].append(tp_rate)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['fpr'].append(system_metrics["false_positive_rate"])

    # Constants for theoretical calculations
    theoretical_fpr = 2**-8  # 0.00390625 for gf_exp=8
    
    # Create dense x-axis for smoother theoretical curves
    tp_rates_dense = np.linspace(0, 1, 100)
    
    # Calculate theoretical reliability using the formula: reliability = 1 - (1-p) × FPR
    theoretical_reliability = [1 - (1-p) * theoretical_fpr for p in tp_rates_dense]

    # Create reliability plot with all systems and theoretical curve
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot empirical data for each system
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['tp_rates'], results['reliability'], 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)],
                label=f"{system_name} (empirical)",
                linewidth=2,
                markersize=6)
    
    # Add theoretical reliability curve
    plt.plot(tp_rates_dense, theoretical_reliability, 
             linestyle='--', color='black', 
             label=f"Theoretical (FPR = 2^-8 ≈ {theoretical_fpr:.6f})",
             linewidth=2)
    
    plt.title(f'Reliability vs True Positive Rate - {num_trials:,} trials', fontsize=14, fontweight='bold')
    plt.xlabel('True Positive Rate', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_true_positives)
    plt.tight_layout()
    plt.savefig('analyses/true_positive_rate_influence/reliability_comparison.png', dpi=300, bbox_inches='tight')

    # Create false positive rate plot with all systems and theoretical line
    plt.figure(figsize=(12, 6))
    
    # Plot empirical FPR for each system
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['tp_rates'], results['fpr'], 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)],
                label=f"{system_name} (empirical)",
                linewidth=2,
                markersize=6)
    
    # Add theoretical FPR (constant)
    plt.plot([0, 1], [theoretical_fpr, theoretical_fpr], 
             linestyle='--', color='black', 
             label=f"Theoretical FPR = 2^-8 ≈ {theoretical_fpr:.6f}",
             linewidth=2)
    
    plt.title(f'False Positive Rate vs True Positive Rate - {num_trials:,} trials', fontsize=14, fontweight='bold')
    plt.xlabel('True Positive Rate', fontsize=12)
    plt.ylabel('False Positive Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_true_positives)
    plt.tight_layout()
    plt.savefig('analyses/true_positive_rate_influence/fpr_comparison.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Theoretical FPR = 2^-8 ≈ {theoretical_fpr:.8f}")
    
    for system_name, results in system_results.items():
        avg_reliability = np.mean(results['reliability'])
        reliability_slopes = np.diff(results['reliability']) / np.diff(results['tp_rates'])
        avg_reliability_slope = np.mean(reliability_slopes) if len(reliability_slopes) > 0 else 0.0
        print(f"{system_name:>8}: Avg Reliability = {avg_reliability:.4f}, "
              f"Avg Reliability Slope = {avg_reliability_slope:.6f}")
        avg_fpr = np.mean([fpr for fpr in results['fpr'] if fpr != 0.0])
        print(f"{system_name:>8}: Avg Reliability = {avg_reliability:.4f}, Avg FPR = {avg_fpr:.6f}")

if __name__ == "__main__":
    main()

# OUTPUT:
# ============================================================
# SUMMARY STATISTICS
# ============================================================
# Theoretical FPR = 2^-8 ≈ 0.00390625
#     RSID: Avg Reliability = 0.9980, Avg Reliability Slope = 0.0039
#     RSID: Avg Reliability = 0.9980, Avg FPR = 0.003921
#     RMID: Avg Reliability = 0.9980, Avg Reliability Slope = 0.0039
#     RMID: Avg Reliability = 0.9980, Avg FPR = 0.003961
#   SHA1ID: Avg Reliability = 0.9980, Avg Reliability Slope = 0.0039
#   SHA1ID: Avg Reliability = 0.9980, Avg FPR = 0.003981