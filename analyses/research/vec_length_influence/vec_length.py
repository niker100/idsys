"""Test for examining the influence of the vec_length parameter on reliability, execution time, and code rate
"""
from idsys import IdMetrics, create_id_system
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - VEC_LENGTH INFLUENCE")
    print("=" * 50)

    # Range of vec_length values to test
    vec_lengths = [1,2,3,4,5,6]

    # Create systems as a dictionary for compare_systems
    systems = {
        "RSID": lambda vec_len: create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]}),
        "RMID": lambda vec_len: create_id_system("RMID", {"gf_exp": 8, "tag_pos": [2]}),
        "SHA1ID": lambda vec_len: create_id_system("SHA1ID", {"gf_exp": 8})
    }

    # Store results for each system
    system_results = {name: {'vec_length': [], 'false_positive_rate': [], 'exec_time': []} for name in systems.keys()}

    for vec_len in vec_lengths:
        print(f"\nEvaluating with vec_length: {vec_len}")
        system_instances = {name: make_sys(vec_len) for name, make_sys in systems.items()}
        metrics = IdMetrics.compare_systems(
            system_instances,
            num_messages=5000000,
            vec_len=vec_len
        )
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['vec_length'].append(vec_len)
            system_results[system_name]['false_positive_rate'].append(system_metrics["false_positive_rate"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])

    # Plot reliability vs vec_length
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['vec_length'], results['false_positive_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Reliability vs vec_length - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('vec_length', fontsize=12)
    plt.ylabel('false_positive_rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(vec_lengths)
    plt.tight_layout()
    plt.savefig('analyses/vec_length_influence/false_positive_rate_vs_vec_length.png', dpi=300, bbox_inches='tight')

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
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.xticks(vec_lengths)
    plt.tight_layout()
    plt.savefig('analyses/vec_length_influence/exec_time_vs_vec_length.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for system_name, results in system_results.items():
        avg_false_positive_rate = np.mean(results['false_positive_rate'])
        avg_exec_time = np.mean(results['exec_time'])
        avg_code_rate = np.mean(results['code_rate'])
        print(f"{system_name:>8}: Avg false_positive_rate = {avg_false_positive_rate:.4f}, Avg Exec Time = {avg_exec_time:.3f} ms, Avg Code Rate = {avg_code_rate:.3f}")

if __name__ == "__main__":
    main()
