"""Test for examining the influence of the true positive rate on the reliability of the identification
"""
from framework import IdMetrics
from framework import create_id_system, generate_test_messages
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
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=100)

    p_true_positives = [i / 100 for i in range(0, 101, 5)]  # True positive rates from 0.0 to 1.0
    
    # Store results for each system
    system_results = {name: {'tp_rates': [], 'reliability': [], 'fpr': []} for name in systems.keys()}

    for tp_rate in p_true_positives:
        print(f"\nEvaluating with True Positive Rate: {tp_rate}")

        # Evaluate all systems with current true positive rate
        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            num_trials=5000000,
            timing_iterations=0,
            p_true_positive=tp_rate
        )

        # Store results for each system
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['tp_rates'].append(tp_rate)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['fpr'].append(system_metrics["false_positive_rate"])

    # Create reliability plot with all systems
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['tp_rates'], results['reliability'], 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)],
                label=system_name,
                linewidth=2,
                markersize=6)
    
    plt.title('Reliability vs True Positive Rate - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('True Positive Rate', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_true_positives)
    plt.tight_layout()
    plt.savefig('analyses/true_positive_rate_influence/reliability_comparison.png', dpi=300, bbox_inches='tight')

    # Create false positive rate plot with all systems
    plt.figure(figsize=(12, 6))
    
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['tp_rates'], results['fpr'], 
                marker=markers[i % len(markers)], 
                color=colors[i % len(colors)],
                label=system_name,
                linewidth=2,
                markersize=6)
    
    plt.title('False Positive Rate vs True Positive Rate - System Comparison', fontsize=14, fontweight='bold')
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
    
    for system_name, results in system_results.items():
        avg_reliability = np.mean(results['reliability'])
        avg_fpr = np.mean(results['fpr'])
        print(f"{system_name:>8}: Avg Reliability = {avg_reliability:.4f}, Avg FPR = {avg_fpr:.4f}")

if __name__ == "__main__":
    main()