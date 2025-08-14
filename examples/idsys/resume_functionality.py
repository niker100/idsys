"""
Minimal example demonstrating the checkpoint/resume functionality in IDSYS.

This example shows:
1. Creating a checkpoint manager
2. Running analysis with automatic saves
3. Simulating an interruption
4. Resuming the analysis from where it left off
"""

import os
import time
import shutil
from idsys import create_id_system, create_checkpoint_manager

def run_simple_analysis(params):
    """Simple analysis function that returns some metrics."""
    print(f"  Analyzing: {params}")
    
    # Create a system based on parameters
    system = create_id_system(
        params["system_type"], 
        {"gf_exp": params["gf_exp"]}
    )
    
    # Simulate analysis taking time
    time.sleep(1)
    
    # Return simulated metrics
    return {
        "execution_time": 10 * params["gf_exp"],
        "false_positive_rate": 1.0 / (2 ** params["gf_exp"]),
        "throughput": 1000 / params["gf_exp"]
    }

def main():
    # Clean up any existing checkpoint directory first
    output_dir = "examples/checkpoint_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned up existing checkpoint directory: {output_dir}")
    
    # Define parameters to test
    parameter_sets = []
    for system_type in ["RSID", "SHA1ID", "RMID"]:
        for gf_exp in [8, 16, 32]:
            parameter_sets.append({
                "system_type": system_type,
                "gf_exp": gf_exp
            })
    
    total_params = len(parameter_sets)
    print(f"Total parameter combinations: {total_params}")
    
    # PART 1: Start analysis and process only first few combinations
    print("\n=== PART 1: Starting analysis (will interrupt) ===")
    
    # Create a checkpoint manager
    checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="minimal_example",
        save_interval=1  # Save after every parameter
    )
    
    # Initialize the analysis
    remaining_params = checkpoint.initialize_analysis(parameter_sets)
    print(f"Parameters to process: {len(remaining_params)}")
    
    # Process only the first 3 parameter combinations
    max_iterations = 3
    for i, params in enumerate(remaining_params[:max_iterations]):
        print(f"\n[{i+1}/{max_iterations}] Processing parameter set {i+1}")
        
        # Run the analysis
        results = run_simple_analysis(params)
        
        # Add the result to the checkpoint
        checkpoint.add_result(params, results)
        
        # Show progress
        completion = checkpoint.get_completion_percentage()
        print(f"  Results: FP_rate={results['false_positive_rate']:.8f}")
        print(f"  Progress: {completion:.1f}% complete")
    
    # Show current results
    df_partial = checkpoint.get_results_dataframe()
    print(f"\nProcessed {len(df_partial)} parameter combinations")
    print(f"Checkpoint file: {checkpoint.csv_file}")
    
    print("\n*** SIMULATING INTERRUPTION ***")
    print("(In a real scenario, this might be a crash, power outage, etc.)")
    
    # PART 2: Resume the analysis
    print("\n=== PART 2: Resuming analysis ===")
    
    # Create a new checkpoint manager (simulating restarting the script)
    resumed_checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="minimal_example",
        save_interval=1
    )
    
    # Initialize - this will detect and load the previous progress
    remaining_params = resumed_checkpoint.initialize_analysis(parameter_sets)
    print(f"Remaining parameters: {len(remaining_params)}")
    
    # Process all remaining parameters
    for i, params in enumerate(remaining_params):
        print(f"\n[{i+1}/{len(remaining_params)}] Resuming with parameter set")
        
        # Run the analysis
        results = run_simple_analysis(params)
        
        # Add result to checkpoint
        resumed_checkpoint.add_result(params, results)
        
        # Show progress
        completion = resumed_checkpoint.get_completion_percentage()
        print(f"  Results: FP_rate={results['false_positive_rate']:.8f}")
        print(f"  Progress: {completion:.1f}% complete")
    
    # Finalize the analysis
    resumed_checkpoint.finalize_analysis()
    
    # Get the final results
    final_results = resumed_checkpoint.get_results_dataframe()
    print(f"\nFinal results: {len(final_results)} rows (expected: {total_params})")
    
    if len(final_results) == total_params:
        print("✅ SUCCESS: All parameters were processed successfully!")
    else:
        print("❌ ERROR: Some parameters were not processed")
    
    print(f"\nFinal results saved to: {resumed_checkpoint.csv_file}")
    
    # Show how to access results
    print("\nResults preview:")
    print(f"  Columns: {list(final_results.columns)}")
    print(f"  First row: {dict(final_results.iloc[0])}")

if __name__ == "__main__":
    main()