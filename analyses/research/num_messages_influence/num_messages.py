"""Test for examining the influence of the number of messages on reliability, execution time, and code rate
"""

import os
from idsys import IdMetrics, create_id_system
import numpy as np
import json

def main():
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - NUMBER OF MESSAGES INFLUENCE")
    print("=" * 50)

    # Range of vec_length values to test
    #num_messages = [10, 50, 100, 500, 1000, 5000, 10000]
    num_messages = np.arange(500000, 10000001, 500000, dtype=int).tolist()  # From 50000 to 1000000 in steps of 5000
    vec_len = 2  # Fixed vector length for this test
    gf_exp = 8  # Galois Field exponent for the systems

    # Create systems as a dictionary for compare_systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        'RMID': create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 1}),
        'SHA1ID': create_id_system("SHA1ID", {"gf_exp": gf_exp})
    }

    # Store results for each system
    system_results = {name: {'num_msg': [], 'false_positives': [], 'exec_time': [], 'fp_rate': []} for name in systems.keys()}

    for num_msg in num_messages:
        print(f"\nEvaluating with number of messages: {num_msg}")
        
        metrics = IdMetrics.compare_systems(
            systems=systems,
            num_messages=num_msg,
            vec_len=vec_len,
            message_subset_size=0,
        )

        for system_name, system_metrics in metrics.items():
            system_results[system_name]['num_msg'].append(num_msg)
            system_results[system_name]['false_positives'].append(system_metrics["false_positives"])
            system_results[system_name]['exec_time'].append(system_metrics["avg_execution_time_ms"])
            system_results[system_name]['fp_rate'].append(system_metrics["false_positive_rate"])

    # Save system results to a file for analysis in a different script

    output_file = 'analyses/num_messages_influence/system_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Include additional metadata in the results
    results_with_metadata = {
        "vec_len": vec_len,
        "num_messages": num_messages,
        "gf_exp": gf_exp,
        "system_results": system_results
    }

    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=4)

    print(f"System results saved to {output_file}")

    return

   
if __name__ == "__main__":
    main()
