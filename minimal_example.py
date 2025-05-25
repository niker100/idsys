#!/usr/bin/env python3
"""
Minimal example showcasing all functions in the identification systems framework.
"""

import numpy as np
from framework import (
    # Core components
    create_id_system, generate_test_messages, IdMetrics,
    # Utility functions
    evaluate_system_with_generated_messages, batch_evaluate_parameters,
    # Individual encoders (for direct usage)
    RSIDEncoder, RS2IDEncoder, RMIDEncoder, SHA1IDEncoder, SHA256IDEncoder
)

def main():
    print("=" * 60)
    print("IDENTIFICATION SYSTEMS FRAMEWORK - MINIMAL EXAMPLE")
    print("=" * 60)
    
    # ==============================================
    # 1. BASIC SYSTEM CREATION AND USAGE
    # ==============================================
    print("\n1. Basic System Creation and Usage:")
    print("-" * 40)
    
    # Create different types of identification systems
    systems = {
        'RSID': create_id_system("RSID", {"gf_exp": 8, "tag_pos": 2}),
        'RS2ID': create_id_system("RS2ID", {"gf_exp": 8, "tag_pos": 2, "tag_pos_in": 2}),
        'RMID': create_id_system("RMID", {"gf_exp": 8, "tag_pos": 2, "rm_order": 1}),
        'SHA1ID': create_id_system("SHA1ID", {"gf_exp": 8}),
        'SHA256ID': create_id_system("SHA256ID", {"gf_exp": 8})
    }
    
    # Generate test messages
    vec_len = 32
    gf_exp = 8
    num_messages = 10
    
    print(f"Generating {num_messages} test messages (vec_len={vec_len}, gf_exp={gf_exp})...")
    messages = generate_test_messages(vec_len, gf_exp, num_messages)
    print(f"Generated messages: {len(messages)} messages")
    print(f"First message: {messages[0][:10]}... (length: {len(messages[0])})")
    
    # Test basic encoding/decoding
    print("\nTesting basic encoding/decoding:")
    for name, system in list(systems.items())[:2]:  # Test first 2 systems
        try:
            message = messages[0]
            tag = system.send(message)
            verification = system.receive(tag, message)
            print(f"{name}: tag={tag}, verification={verification}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    # ==============================================
    # 2. COMPREHENSIVE SYSTEM EVALUATION
    # ==============================================
    print("\n2. Comprehensive System Evaluation:")
    print("-" * 40)
    
    # Evaluate a single system comprehensively
    test_system = systems['RSID']
    print("Evaluating RSID system...")
    
    try:
        metrics = IdMetrics.evaluate_system(
            test_system, 
            messages, 
            num_trials=100,  # Reduced for quick example
            timing_iterations=50
        )
        
        print("Key metrics:")
        key_metrics = ['reliability', 'false_positive_rate', 'code_rate', 
                      'avg_execution_time_ms', 'computational_efficiency']
        for metric in key_metrics:
            if metric in metrics:
                print(f"  {metric}: {metrics[metric]:.4f}")
        
    except Exception as e:
        print(f"Error evaluating RSID: {e}")
    
    # ==============================================
    # 3. SYSTEM COMPARISON
    # ==============================================
    print("\n3. System Comparison:")
    print("-" * 40)
    
    print("Comparing multiple systems...")
    try:
        # Compare subset of systems for speed
        comparison_systems = {k: v for k, v in list(systems.items())[:3]}
        
        comparison_results = IdMetrics.compare_systems(
            comparison_systems,
            messages,
            num_trials=50,  # Reduced for quick example
            timing_iterations=25
        )
        
        print("\nComparison Results:")
        for system_name, metrics in comparison_results.items():
            print(f"\n{system_name}:")
            key_metrics = ['reliability', 'false_positive_rate', 'code_rate', 'computational_efficiency']
            for metric in key_metrics:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
    
            
    except Exception as e:
        print(f"Error in system comparison: {e}")
    
    # ==============================================
    # 4. UTILITY FUNCTIONS
    # ==============================================
    print("\n4. Utility Functions:")
    print("-" * 40)
    
    # Evaluate system with auto-generated messages
    print("Testing evaluate_system_with_generated_messages...")
    try:
        auto_metrics = evaluate_system_with_generated_messages(
            systems['RSID'],
            vec_len=16,
            gf_exp=8,
            num_messages=20,
            num_trials=50,
            timing_iterations=25
        )
        print(f"Auto-evaluation reliability: {auto_metrics.get('reliability', 'N/A'):.4f}")
        print(f"Auto-evaluation code rate: {auto_metrics.get('code_rate', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error in auto-evaluation: {e}")
    
    # Batch parameter evaluation
    print("\nTesting batch_evaluate_parameters...")
    try:
        parameter_grid = {
            'gf_exp': [6, 8],
            'tag_pos': [1, 2]
        }
        
        batch_results = batch_evaluate_parameters(
            'RSID',
            parameter_grid,
            vec_len=16,
            gf_exp=8,  # For message generation
            num_messages=10,
            num_trials=25,
            timing_iterations=10
        )
        
        print("Batch evaluation results:")
        for param_combo, metrics in batch_results.items():
            reliability = metrics.get('reliability', 0)
            code_rate = metrics.get('code_rate', 0)
            print(f"  {param_combo}: reliability={reliability:.3f}, code_rate={code_rate:.3f}")
            
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
    
    # ==============================================
    # 5. DIRECT ENCODER USAGE
    # ==============================================
    print("\n5. Direct Encoder Usage:")
    print("-" * 40)
    
    print("Testing direct encoder instantiation...")
    try:
        # Create encoder directly
        encoder = RSIDEncoder({"gf_exp": 6, "tag_pos": 1})
        
        # Test with a simple message
        test_message = messages[0][:8]  # Use shorter message
        tag = encoder.encode(test_message)
        print(f"Direct RSID encoding: message length={len(test_message)}, tag={tag}")
        
        # Test parameter updates
        encoder.set_parameters({"tag_pos": 3})
        new_tag = encoder.encode(test_message)
        print(f"After parameter update: new tag={new_tag}")
        
    except Exception as e:
        print(f"Error in direct encoder usage: {e}")
    
    # ==============================================
    # 6. ADVANCED METRICS ANALYSIS
    # ==============================================
    print("\n6. Advanced Metrics Analysis:")
    print("-" * 40)
    
    try:
        # Get detailed metrics for analysis
        detailed_metrics = IdMetrics.evaluate_system(
            systems['RSID'],
            messages[:5],  # Use fewer messages for speed
            num_trials=50,
            timing_iterations=25
        )
        
        print("Detailed metrics overview:")
        metric_categories = {
            'Performance': ['reliability', 'false_positive_rate', 'overall_performance_score'],
            'Efficiency': ['code_rate', 'computational_efficiency', 'throughput_msgs_per_sec'],
            'Information': ['message_entropy', 'tag_entropy', 'compression_ratio'],
            'System': ['tag_size_bits', 'avg_message_length', 'tag_uniqueness']
        }
        
        for category, metrics_list in metric_categories.items():
            print(f"\n  {category} Metrics:")
            for metric in metrics_list:
                if metric in detailed_metrics:
                    value = detailed_metrics[metric]
                    print(f"    {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error in advanced metrics analysis: {e}")
    
    # ==============================================
    # 7. ERROR HANDLING AND EDGE CASES
    # ==============================================
    print("\n7. Error Handling Examples:")
    print("-" * 40)
    
    # Test with empty message set
    try:
        IdMetrics.evaluate_system(systems['RSID'], [], num_trials=10)
    except ValueError as e:
        print(f"✓ Correctly caught empty message set error: {e}")
    
    # Test with invalid system type
    try:
        create_id_system("INVALID_TYPE")
    except ValueError as e:
        print(f"✓ Correctly caught invalid system type: {e}")
    
    # Test with unsupported GF exponent
    try:
        create_id_system("RSID", {"gf_exp": 128})  # Too large
    except Exception as e:
        print(f"✓ Correctly caught unsupported GF exponent: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("MINIMAL EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()