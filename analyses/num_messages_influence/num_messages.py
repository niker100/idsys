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
    num_messages = np.arange(5000, 100001, 5000, dtype=int)
    vec_len = 1250  # Fixed vector length for this test
    trials = 100000  # Number of trials for each system evaluation
    gf_exp = 8  # Galois Field exponent for the systems

    # Create systems as a dictionary for compare_systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": 2}),
        'SHA1ID': create_id_system("SHA1ID", {"gf_exp": gf_exp}),
        'RMID': create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": 2, "rm_order": 1})
    }

    # Store results for each system
    system_results = {name: {'num_msg': [], 'reliability': [], 'false_positives': [], 'exec_time': [], 'code_rate': []} for name in systems.keys()}

    for num_msg in num_messages:
        print(f"\nEvaluating with number of messages: {num_msg}")
        messages = generate_test_messages(vec_len=vec_len, gf_exp=8, count=num_msg)
        
        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            num_trials=trials,
            timing_iterations=0,
            p_true_positive=0
        )

        for system_name, system_metrics in metrics.items():
            system_results[system_name]['num_msg'].append(num_msg)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['false_positives'].append(system_metrics["false_positives"])
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

    # Plot false positives vs number of messages
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['num_msg'], results['false_positives'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('False Positives vs Number of Messages - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('False Positives', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(num_messages)

    # Add textbox with trials, gf_exp, and vec_len
    textstr = f'Trials: {trials}\nGF Exp: {gf_exp}\nVec Len: {vec_len}'
    plt.gcf().text(0.95, 0.5, textstr, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('analyses/num_messages_influence/false_positives_vs_num_messages.png', dpi=300, bbox_inches='tight')

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
