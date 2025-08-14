"""
Parallel processing example for the IDSYS framework.

This example demonstrates how to use parallel processing for 
evaluating identification systems with large message sets.
"""

import time
import os
from idsys import create_id_system, IdMetrics

def main():
    print("== IDSYS Parallel Processing Example ==")
    
    # Get available CPU cores
    cpu_count = os.cpu_count()
    print(f"Available CPU cores: {cpu_count}")
    
    # Create an identification system
    system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
    
    # Compare single vs multi-process performance
    test_configurations = [
        {"num_processes": 1, "label": "Single process"},
        {"num_processes": None, "label": "Auto (multi-process)"}
    ]
    
    message_counts = [1000, 5000]
    
    for msg_count in message_counts:
        print(f"\nTesting with {msg_count} messages:")
        print("-" * 40)
        
        for config in test_configurations:
            processes = config["num_processes"]
            label = config["label"]
            
            print(f"\n{label}:")
            start_time = time.time()
            
            # Run evaluation with specified process count
            metrics = IdMetrics.evaluate_system(
                system,
                num_messages=msg_count,
                vec_len=16,
                num_processes=processes,
                calculate_pdfs=False  # Skip PDF calculation for speed
            )
            
            elapsed_time = time.time() - start_time
            print(f"  Time: {elapsed_time:.2f} seconds")
            print(f"  Messages processed: {metrics.get('total_messages', 0)}")
            print(f"  False positive rate: {metrics.get('false_positive_rate', 0):.8f}")
            print(f"  Average execution time: {metrics.get('avg_execution_time_ms', 0):.3f} ms")
    
    # Demonstrate process count scaling
    print("\nProcess Count Scaling:")
    print("-" * 40)
    
    # Test with different explicit process counts
    process_counts = [1, 2, 4, min(8, max(cpu_count, 1))]
    message_count = 10000  # Use larger count to demonstrate parallelism benefits
    
    for processes in process_counts:
        if processes > cpu_count:
            print(f"Skipping {processes} processes (exceeds available {cpu_count} cores)")
            continue
            
        print(f"\nProcesses: {processes}")
        start_time = time.time()
        
        # Run evaluation with specified process count
        metrics = IdMetrics.evaluate_system(
            system,
            num_messages=message_count,
            vec_len=16,
            num_processes=processes,
            calculate_pdfs=False
        )
        
        elapsed_time = time.time() - start_time
        print(f"  Time: {elapsed_time:.2f} seconds")
        
    print("\nNote: Optimal process count depends on your CPU and workload")
    print("For CPU-bound tasks, using num_processes=None is usually best")

if __name__ == "__main__":
    main()