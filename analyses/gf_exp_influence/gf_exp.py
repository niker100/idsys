"""Test for examining the influence of the gf_exp parameter on reliability and execution time
"""
from framework import IdMetrics
from framework import create_id_system, generate_test_messages
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - GF_EXP INFLUENCE")
    print("=" * 50)

    # Range of gf_exp values to test
    gf_exp_values = [8, 16, 32, 64]

    # Create systems as a dictionary for compare_systems
    system_types = [
        ("RSID", lambda gf_exp: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": 2})),
        ("RMID", lambda gf_exp: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": 2})),
        ("SHA1ID", lambda gf_exp: create_id_system("SHA1ID", {"gf_exp": gf_exp}))
    ]

    # Store results for each system
    system_results = {name: {'gf_exp': [], 'reliability': [], 'exec_time': [], 'fp_rate': []} for name, _ in system_types}

    for gf_exp in gf_exp_values:
        print(f"\nEvaluating with GF_EXP: {gf_exp}")
        # Generate test messages for this gf_exp
        messages = generate_test_messages(vec_len=8, gf_exp=gf_exp, count=1000)
        systems = {name: make_sys(gf_exp) for name, make_sys in system_types}

        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            #num_trials=100000,  # Fewer trials for speed
            num_trials=2**18,  # Fewer trials for speed
            timing_iterations=1000,
            p_true_positive=0.5
        )

        # Store results for each system
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['gf_exp'].append(gf_exp)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])
            system_results[system_name]['fp_rate'].append(system_metrics["false_positive_rate"])

    #TODO: Analyze different time metrics

    # Plot reliability vs gf_exp
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['reliability'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Reliability vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)
    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/reliability_vs_gf_exp.png', dpi=300, bbox_inches='tight')

    # Plot execution time vs gf_exp
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['exec_time'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Execution Time vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)
    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/exec_time_vs_gf_exp.png', dpi=300, bbox_inches='tight')

    # Plot false positive rate vs gf_exp
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['gf_exp'], results['fp_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positive Rate vs GF_EXP - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('GF_EXP', fontsize=12)
    plt.ylabel('False Positive Rate', fontsize=12)
    plt.grid(True, alpha=0.3)    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(gf_exp_values)
    plt.tight_layout()
    plt.savefig('analyses/gf_exp_influence/fp_rate_vs_gf_exp.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for system_name, results in system_results.items():
        avg_reliability = np.mean(results['reliability'])
        avg_time = np.mean(results['exec_time'])
        print(f"{system_name:>8}: Avg Reliability = {avg_reliability:.4f}, Avg Exec Time = {avg_time:.3f}s")

if __name__ == "__main__":
    main()
