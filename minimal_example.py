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
    print("=" * 50)
    print("IDENTIFICATION SYSTEMS - MINIMAL EXAMPLE")
    print("=" * 50)
    
    # ==============================================
    # 1. BASIC SYSTEM SETUP
    # ==============================================
    print("\n1. Creating Systems and Test Data:")
    print("-" * 30)
    
    # Create systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2}),
        'SHA1ID': create_id_system("SHA1ID", {"gf_exp": 8}),
        'RMID': create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2, "rm_order": 1})
    }
    
    # Generate test messages
    messages = generate_test_messages(vec_len=32, gf_exp=8, count=10)
    print(f"Generated {len(messages)} test messages")
    
    # ==============================================
    # 2. BASIC ENCODING/DECODING
    # ==============================================
    print("\n2. Basic Encoding/Decoding:")
    print("-" * 30)
    
    for name, system in systems.items():
        try:
            message = messages[0]
            tag = system.send(message)
            verification = system.receive(tag, message)
            print(f"{name}: tag={tag}, verified={verification}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    # ==============================================
    # 3. SYSTEM EVALUATION
    # ==============================================
    print("\n3. System Evaluation:")
    print("-" * 30)
    
    # Evaluate RSID system
    try:
        metrics = IdMetrics.evaluate_system(
            systems['RSID'], 
            messages[:5],  # Use subset for speed
            num_trials=50,
            timing_iterations=25
        )
        
        print("RSID Metrics:")
        key_metrics = ['reliability', 'false_positive_rate', 'code_rate', 'computational_efficiency']
        for metric in key_metrics:
            if metric in metrics:
                print(f"  {metric}: {metrics[metric]:.4f}")
        
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    # ==============================================
    # 4. SYSTEM COMPARISON
    # ==============================================
    print("\n4. System Comparison:")
    print("-" * 30)
    
    try:
        comparison = IdMetrics.compare_systems(
            systems,
            messages[:5],
            num_trials=25,
            timing_iterations=10
        )
        
        print("Comparison Results:")
        for system_name, metrics in comparison.items():
            reliability = metrics.get('reliability', 0)
            code_rate = metrics.get('code_rate', 0)
            print(f"  {system_name}: reliability={reliability:.3f}, code_rate={code_rate:.3f}")
            
    except Exception as e:
        print(f"Comparison error: {e}")
    
    # ==============================================
    # 5. UTILITY FUNCTION
    # ==============================================
    print("\n5. Auto-Evaluation Utility:")
    print("-" * 30)
    
    try:
        auto_metrics = evaluate_system_with_generated_messages(
            systems['RSID'],
            vec_len=16,
            num_messages=10,
            num_trials=25
        )
        print(f"Auto-evaluation reliability: {auto_metrics.get('reliability', 0):.4f}")
        
    except Exception as e:
        print(f"Auto-evaluation error: {e}")
    
    # ==============================================
    # 6. DIRECT ENCODER USAGE
    # ==============================================
    print("\n6. Direct Encoder Usage:")
    print("-" * 30)
    
    try:
        encoder = RSIDEncoder({"gf_exp": 6, "tag_pos": 1})
        test_message = messages[0][:8]
        tag = encoder.encode(test_message)
        print(f"Direct encoding: message_len={len(test_message)}, tag={tag}")
        
    except Exception as e:
        print(f"Direct encoder error: {e}")    

    # ==============================================
    # 7. Batch Evaluation Example
    # ==============================================
    print("\n7. Batch Evaluation Example:")
    print("-" * 30)

    try:
        param_grid = {
            'gf_exp': [6, 8],
            'tag_pos': [1, 2],
            'rm_order': [1, 2]
        }
        
        batch_results = batch_evaluate_parameters(
            systems['RMID'], 
            param_grid, 
            vec_len=16,
            num_messages=100,
            num_trials=10
        )
        
        print("Batch Evaluation Results:")
        for params, metrics in batch_results.items():
            print(f"  {params}: reliability={metrics.get('reliability', 0):.4f}, code_rate={metrics.get('code_rate', 0):.4f}")
    except Exception as e:
        print(f"Batch evaluation error: {e}")

    print("\n" + "=" * 50)
    print("MINIMAL EXAMPLE COMPLETED!")
    print("=" * 50)

if __name__ == "__main__":
    main()