"""Test for examining the influence of the gf_exp parameter on reliability and execution time
"""
import os
from idsys import IdMetrics, create_id_system
import json

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - GF_EXP INFLUENCE")
    print("=" * 50)

    # Range of gf_exp values to test
    gf_exp_values = [8, 16, 32]
    num_messages = 10**8  # Number of messages to generate for each gf_exp
    vec_len = 8  # Fixed vector length for this test 1000bit

    # Create systems as a dictionary for compare_systems
    system_types = [
        ("RSID", lambda gf_exp: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]})),
        ("RMID", lambda gf_exp: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2]})),
        ("SHA1ID", lambda gf_exp: create_id_system("SHA1ID", {"gf_exp": gf_exp}))
    ]

    # Store results for each system
    system_results = {name: {'gf_exp': [], 'false_positives': [], 'exec_time': [], 'fp_rate': []} for name, _ in system_types}

    for gf_exp in gf_exp_values:
        print(f"\nEvaluating with GF_EXP: {gf_exp}")
        systems = {name: make_sys(gf_exp) for name, make_sys in system_types}

        metrics = IdMetrics.compare_systems(
            systems=systems,
            num_messages=num_messages,
            vec_len=vec_len,
        )

        # Store results for each system
        for system_name, system_metrics in metrics.items():
            system_results[system_name]['gf_exp'].append(gf_exp)
            system_results[system_name]['false_positives'].append(system_metrics["false_positives"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])
            system_results[system_name]['fp_rate'].append(system_metrics["false_positive_rate"])

    #TODO: Analyze different time metrics

    # Save system results to a file for analysis in a different script

    output_file = 'analyses/gf_exp_influence/system_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Include additional metadata in the results
    results_with_metadata = {
        "vec_len": vec_len,
        "num_messages": num_messages,
        "gf_exp_values": gf_exp_values,
        "system_results": system_results
    }

    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=4)

    print(f"System results saved to {output_file}")

    return    

if __name__ == "__main__":
    main()

