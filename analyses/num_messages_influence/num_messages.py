"""Test for examining the influence of the number of messages on reliability, execution time, and code rate
"""

import sys
import os
#Add path to parent directory to import framework modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from framework import IdMetrics
from framework import create_id_system, generate_test_messages
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - NUMBER OF MESSAGES INFLUENCE")
    print("=" * 50)

    # Range of vec_length values to test
    #num_messages = [10, 50, 100, 500, 1000, 5000, 10000]
    num_messages = np.linspace(1000, 10000, num=19, dtype=int)

    # Create systems as a dictionary for compare_systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2}),
        'SHA1ID': create_id_system("SHA1ID", {"gf_exp": 8}),
        'RMID': create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2, "rm_order": 1})
    }

    # Store results for each system
    system_results = {name: {'num_msg': [], 'reliability': [], 'exec_time': [], 'code_rate': []} for name in systems.keys()}

    for num_msg in num_messages:
        print(f"\nEvaluating with number of messages: {num_msg}")
        messages = generate_test_messages(vec_len=32, gf_exp=8, count=num_msg)
        
        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            num_trials=10000,
            timing_iterations=1000,
            p_true_positive=0.5
        )

        for system_name, system_metrics in metrics.items():
            system_results[system_name]['num_msg'].append(num_msg)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])
            system_results[system_name]['code_rate'].append(system_metrics["code_rate"])


    # Plot reliability vs number of messages
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['reliability'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Reliability vs number of messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('number of messages', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)
    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/reliability_vs_num_messages.png', dpi=300, bbox_inches='tight')


    # Plot execution time vs number of messages
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['exec_time'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Execution Time vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('Avg Execution Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)
    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/exec_time_vs_num_messages.png', dpi=300, bbox_inches='tight')

    # Plot code rate vs number of messages
    plt.figure(figsize=(12, 6))
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['code_rate'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Code Rate vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('Code Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)
    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/code_rate_vs_num_messages.png', dpi=300, bbox_inches='tight')

   
if __name__ == "__main__":
    main()
