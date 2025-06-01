"""Test for examining the influence of the vec_length parameter on reliability, execution time, and code rate
"""
from framework import IdMetrics
from framework import create_id_system, generate_test_messages
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - VEC_LENGTH INFLUENCE")
    print("=" * 50)

    # Range of vec_length values to test
    vec_lengths = [4, 8, 16, 32, 64, 128]
    #vec_lengths = [10, 30, 50, 70, 90, 110]

    # Create systems as a dictionary for compare_systems
    systems = {
        "RSID": lambda vec_len: create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2, "vec_len": vec_len}),
        "RMID": lambda vec_len: create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2, "vec_len": vec_len}),
        "SHA1ID": lambda vec_len: create_id_system("SHA1ID", {"gf_exp": 8, "vec_len": vec_len})
    }

    # Store results for each system
    system_results = {name: {'vec_length': [], 'reliability': [], 'exec_time': [], 'code_rate': []} for name in systems.keys()}

    for vec_len in vec_lengths:
        print(f"\nEvaluating with vec_length: {vec_len}")
        messages = generate_test_messages(vec_len=vec_len, gf_exp=8, count=100)
        system_instances = {name: make_sys(vec_len) for name, make_sys in systems.items()}
        metrics = IdMetrics.compare_systems(
            system_instances,
            messages,
            num_trials=10000,
            timing_iterations=1000,
            p_true_positive=0.5
        )
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['vec_length'].append(vec_len)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])
            system_results[system_name]['code_rate'].append(system_metrics["code_rate"])

    # Plot reliability vs vec_length
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['vec_length'], results['reliability'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Reliability vs vec_length - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('vec_length', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(vec_lengths)
    plt.tight_layout()
    plt.savefig('analyses/vec_length_influence/reliability_vs_vec_length.png', dpi=300, bbox_inches='tight')

    # Plot execution time vs vec_length
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['vec_length'], results['exec_time'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Execution Time vs vec_length - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('vec_length', fontsize=12)
    plt.ylabel('Avg Execution Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(vec_lengths)
    plt.tight_layout()
    plt.savefig('analyses/vec_length_influence/exec_time_vs_vec_length.png', dpi=300, bbox_inches='tight')

    # Plot code rate vs vec_length
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['vec_length'], results['code_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Code Rate vs vec_length - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('vec_length', fontsize=12)
    plt.ylabel('Code Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(vec_lengths)
    plt.tight_layout()
    plt.savefig('analyses/vec_length_influence/code_rate_vs_vec_length.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for system_name, results in system_results.items():
        avg_reliability = np.mean(results['reliability'])
        avg_exec_time = np.mean(results['exec_time'])
        avg_code_rate = np.mean(results['code_rate'])
        print(f"{system_name:>8}: Avg Reliability = {avg_reliability:.4f}, Avg Exec Time = {avg_exec_time:.3f} ms, Avg Code Rate = {avg_code_rate:.3f}")

if __name__ == "__main__":
    main()
