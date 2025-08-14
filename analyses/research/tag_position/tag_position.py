"""Test for examining the influence of the tag position on reliability (and execution time) for tagging codes

    Expectation: slight influence on reliability and not on execution time
"""
from idsys import IdMetrics, create_id_system, generate_structured_messages
import matplotlib.pyplot as plt

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - Tag position influence")
    print("=" * 50)

    gf_exp = 8
    # Range of tag positions to test
    tag_positions = [x for x in range(0,3)] # TODO Länge noch zu bestimmen

    # Create systems as a dictionary for compare_systems
    system_types = [
        ("RSID", lambda tag_pos: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": tag_pos})),
        ("RMID", lambda tag_pos: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": tag_pos})),
        ]
    
    system_types_random = [
        ("RSID", lambda tag_pos: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": tag_pos})),
        ("RMID", lambda tag_pos: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": tag_pos})),
        ]

    # Store results for each system
    system_results = {name: {'tag_position': [], 'reliability': [], 'exec_time': []} for name, _ in system_types}

    # Generate test messages
    messages = generate_test_messages(vec_len=16, gf_exp=gf_exp, count=100)

    #iterate through all tag positions (deterministic)
    for tag_pos in tag_positions:
        print(f"\nEvaluating with Tag position: {tag_pos}")
        systems = {name: make_sys(tag_pos) for name, make_sys in system_types}

        metrics = IdMetrics.compare_systems(
            systems,
            messages,
            #num_trials=100000,  # Fewer trials for speed
            num_trials=100,  # Fewer trials for speed
            timing_iterations=3,
            p_true_positive=0.5
        )

        # Store results for each system
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['tag_position'].append(tag_pos)
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])

    '''#evaluate system for random tag position
    print(f"\nEvaluating with Random tag position")
    tag_positions_random = np.random.randint(0,3, size = 100)
    systems_random = {name: make_sys(tag_positions_random) for name, make_sys in system_types_random}

    metrics_random = IdMetrics.compare_systems(
            systems_random,
            messages,
            #num_trials=100000,  # Fewer trials for speed
            num_trials=100,  # Fewer trials for speed
            timing_iterations=3,
            p_true_positive=0.5
        )
    
    for system_name, system_metrics in metrics_random.items():
            system_results[system_name]['tag_position'].append(-1) #for random positioning
            system_results[system_name]['reliability'].append(system_metrics["reliability"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])'''

    # Plot reliability vs tag position
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    for i, (system_name, results) in enumerate(system_results.items()):
        plt.plot(results['tag_position'], results['reliability'],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=system_name,
                 linewidth=2,
                 markersize=6)
    plt.title('Reliability vs tag position - System Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('tag position', fontsize=12)
    plt.ylabel('Reliability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(tag_positions)
    plt.tight_layout()
    plt.show()
    plt.savefig('analyses/tag_position/reliability_vs_tag_position.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
