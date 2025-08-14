#!/usr/bin/env python3
"""
Minimal example showcasing core functions in the identification systems framework.
"""

import numpy as np
from framework import (
    create_id_system, generate_test_messages, IdMetrics,
    evaluate_system_with_generated_messages, RSIDEncoder,
    batch_evaluate_parameters
)

def main():
    
    # Create systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": 16, "tag_pos": 2}),
        'RS2ID': create_id_system("RS2ID", {"gf_exp": 16, "tag_pos": 2, "tag_pos_in": 2}),
    }
    
    # Generate test messages
    num_messages = 10**6
    messages = generate_test_messages(vec_len=8, gf_exp=16, count=num_messages)
    print(f"Generated {len(messages)} test messages")

    
    metrics = IdMetrics.evaluate_system(
        systems['RSID'], 
        messages,
        num_trials=0,
        timing_iterations=0,
        max_messages=num_messages
    )
    
    print("RSID Metrics:")
    key_metrics = ['unique_tags', 'tag_entropy', 'tag_distribution_uniformity', 'tag_max_value']
    for metric in key_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")

    metrics = IdMetrics.evaluate_system(
        systems['RS2ID'], 
        messages,
        num_trials=0,
        timing_iterations=0,
        max_messages=num_messages
    )
    print("RS2ID Metrics:")
    for metric in key_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")



if __name__ == "__main__":
    main()