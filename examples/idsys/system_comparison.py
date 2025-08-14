"""
System comparison example for the IDSYS framework.

This example demonstrates creating and comparing multiple
identification systems using various metrics.
"""

from idsys import create_id_system, IdMetrics, generate_test_messages

def main():
    print("== IDSYS System Comparison Example ==")
    
    # Create multiple identification systems
    systems = {
        "RSID": create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]}),
        "SHA1ID": create_id_system("SHA1ID", {"gf_exp": 8}),
        "RMID": create_id_system("RMID", {"gf_exp": 8, "tag_pos": [2], "rm_order": 1}),
    }
    
    print(f"Comparing {len(systems)} systems: {', '.join(systems.keys())}")
    
    # Generate a small set of test messages for comparison
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=100)
    
    # Compare systems one by one for demonstration
    print("\nIndividual System Metrics:")
    print("-" * 40)
    
    for name, system in systems.items():
        print(f"\nEvaluating {name}...")
        
        # Time a single message processing
        import time
        message = messages[0]
        
        start_time = time.perf_counter()
        tag = system.send(message)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        print(f"  Single message encoding time: {execution_time_ms:.3f} ms")
        
        # Get comprehensive metrics (using a smaller sample for speed)
        metrics = IdMetrics.evaluate_system(
            system, 
            num_messages=50,  # Small sample for quick example
            vec_len=16,
            calculate_pdfs=False  # Skip PDF calculation for speed
        )
        
        # Print key metrics
        print(f"  False positive rate: {metrics['false_positive_rate']:.8f}")
        print(f"  Code rate: {metrics['code_rate_bulk']:.6f}")
        print(f"  Avg execution time: {metrics['avg_execution_time_ms']:.3f} ms")
        print(f"  Tag size: {metrics['tag_size_bits']} bits")
    
    # Compare all systems together (batch processing)
    print("\nBatch Comparison Results:")
    print("-" * 40)
    
    try:
        # This simulates the compare_systems utility (for clearer example)
        results = {}
        for name, system in systems.items():
            results[name] = IdMetrics.evaluate_system(
                system,
                num_messages=25,  # Small sample for quick example 
                vec_len=16,
                calculate_pdfs=False
            )
            
        # Create comparison table of key metrics
        metrics_to_compare = [
            'false_positive_rate', 
            'avg_execution_time_ms', 
            'code_rate_bulk',
            'throughput_msgs_per_sec'
        ]
        
        # Print header
        header = "System".ljust(10)
        for metric in metrics_to_compare:
            header += f" | {metric[:10].ljust(10)}"
        print(header)
        print("-" * len(header))
        
        # Print each system's metrics
        for name, metrics in results.items():
            row = name.ljust(10)
            for metric in metrics_to_compare:
                value = metrics.get(metric, 0)
                
                # Format based on metric type
                if 'rate' in metric:
                    formatted = f"{value:.8f}"[:10].ljust(10)
                elif 'time' in metric:
                    formatted = f"{value:.3f}".ljust(10)
                else:
                    formatted = f"{value:.4f}".ljust(10)
                    
                row += f" | {formatted}"
            print(row)
            
    except Exception as e:
        print(f"Comparison error: {e}")

if __name__ == "__main__":
    main()