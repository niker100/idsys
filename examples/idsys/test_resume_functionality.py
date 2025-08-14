#!/usr/bin/env python3
"""
Test script to demonstrate the resume functionality of the checkpointing system.

This script will:
1. Start an analysis 
2. Simulate interruption after a few iterations
3. Resume the analysis from where it left off
"""

import sys
import os
import time

# Add path to parent directory to import framework modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework import IdMetrics, create_id_system
from framework.checkpoint import create_checkpoint_manager


def test_resume_functionality():
    """Test the resume functionality by simulating interruption."""
    
    print("=" * 60)
    print("TESTING RESUME FUNCTIONALITY")
    print("=" * 60)
    
    # Create a larger parameter set for testing
    parameter_sets = []
    gf_exps = [8, 16]
    system_types = ["RSID", "RMID", "SHA1ID"]
    vec_lens = [32, 64]
    
    for gf_exp in gf_exps:
        for system_type in system_types:
            for vec_len in vec_lens:
                parameter_sets.append({
                    "gf_exp": gf_exp,
                    "system_type": system_type,
                    "vec_len": vec_len,
                    "num_messages": 10000  # Smaller for faster testing
                })
    
    print(f"Total parameter combinations: {len(parameter_sets)}")
    
    # PART 1: Start analysis and process only first few combinations
    print("\n--- PART 1: Starting analysis (will simulate interruption) ---")
    
    checkpoint = create_checkpoint_manager(
        output_dir="analyses/multi_tag_multi_message/checkpoints_test",
        analysis_name="resume_test",
        save_interval=1
    )
    
    remaining_params = checkpoint.initialize_analysis(parameter_sets)
    print(f"Remaining parameters: {len(remaining_params)}")
    
    # Process only the first 3 combinations, then "crash"
    max_iterations = 3
    for i, params in enumerate(remaining_params[:max_iterations]):
        print(f"\n[{i+1}/{max_iterations}] Processing: {params}")
        
        # Simulate analysis
        system = create_id_system(params["system_type"], {"gf_exp": params["gf_exp"]})
        systems = {params["system_type"]: system}
        
        metrics = IdMetrics.compare_systems(
            systems=systems,
            num_messages=params["num_messages"],
            vec_len=params["vec_len"],
            message_subset_size=0
        )
        
        results = {
            "false_positive_rate": metrics[params["system_type"]]["false_positive_rate"],
            "false_positives": metrics[params["system_type"]]["false_positives"],
            "avg_execution_time_ms": metrics[params["system_type"]]["avg_execution_time_ms"]
        }
        
        checkpoint.add_result(params, results)
        
        completion = checkpoint.get_completion_percentage()
        print(f"   Results: FP_rate={results['false_positive_rate']:.6f}")
        print(f"   Progress: {completion:.1f}% complete")
    
    print(f"\n!!! SIMULATING CRASH AFTER {max_iterations} ITERATIONS !!!")
    print("(In real scenario, this is where your script would crash)")
    
    # Get current state
    df_before_crash = checkpoint.get_results_dataframe()
    print(f"Results before 'crash': {len(df_before_crash)} rows")
    
    # PART 2: Resume analysis
    print(f"\n--- PART 2: Resuming analysis after 'crash' ---")
    print("Creating new checkpoint manager (simulating script restart)")
    
    # Create a new checkpoint manager (simulating script restart)
    checkpoint_resumed = create_checkpoint_manager(
        output_dir="analyses/multi_tag_multi_message/checkpoints_test",
        analysis_name="resume_test",
        save_interval=1
    )
    
    # This should load the existing checkpoint
    remaining_params_after_resume = checkpoint_resumed.initialize_analysis(parameter_sets)
    print(f"Remaining parameters after resume: {len(remaining_params_after_resume)}")
    
    # Process the remaining combinations
    for i, params in enumerate(remaining_params_after_resume):
        print(f"\n[{i+1}/{len(remaining_params_after_resume)}] Resuming: {params}")
        
        # Simulate analysis
        system = create_id_system(params["system_type"], {"gf_exp": params["gf_exp"]})
        systems = {params["system_type"]: system}
        
        metrics = IdMetrics.compare_systems(
            systems=systems,
            num_messages=params["num_messages"],
            vec_len=params["vec_len"],
            message_subset_size=0
        )
        
        results = {
            "false_positive_rate": metrics[params["system_type"]]["false_positive_rate"],
            "false_positives": metrics[params["system_type"]]["false_positives"],
            "avg_execution_time_ms": metrics[params["system_type"]]["avg_execution_time_ms"]
        }
        
        checkpoint_resumed.add_result(params, results)
        
        completion = checkpoint_resumed.get_completion_percentage()
        print(f"   Results: FP_rate={results['false_positive_rate']:.6f}")
        print(f"   Progress: {completion:.1f}% complete")
    
    # Finalize
    checkpoint_resumed.finalize_analysis()
    
    # PART 3: Verify results
    print(f"\n--- PART 3: Verification ---")
    
    df_final = checkpoint_resumed.get_results_dataframe()
    print(f"Final results: {len(df_final)} rows")
    print(f"Expected total: {len(parameter_sets)} rows")
    
    if len(df_final) == len(parameter_sets):
        print("✅ SUCCESS: All parameter combinations were processed exactly once!")
    else:
        print("❌ ERROR: Mismatch in expected vs actual results")
    
    print(f"\nFinal CSV file: {checkpoint_resumed.csv_file}")
    
    return df_final


if __name__ == "__main__":
    # Clean up any existing test files first
    import shutil
    test_dir = "analyses/multi_tag_multi_message/checkpoints_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Cleaned up existing test directory: {test_dir}")
    
    # Run the test
    results = test_resume_functionality()
    print(f"\nTest completed! Final dataset has {len(results)} rows.")
