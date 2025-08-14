#!/usr/bin/env python3
"""
Enhanced version of gf_exp analysis with checkpointing support.

This script evaluates the influence of the gf_exp parameter on reliability and execution time
for different identification system types.

Features:
- Automatic periodic saving to CSV
- Resume capability after crashes
- Progress tracking
- Backup creation
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from idsys import IdMetrics, create_id_system, create_checkpoint_manager


# Analysis configuration - centralized to avoid duplication
ANALYSIS_CONFIG = {
    "gf_exp_values": [8, 16, 32, 64],
    "num_messages": 100,
    "vec_len": 125,
    "system_types": ["RSID", "RMID", "SHA1ID"],
    "message_subset_size": 0
}


def create_parameter_sets():
    """Generate all parameter combinations to test."""
    parameter_sets = []
    
    # Use centralized configuration
    gf_exp_values = ANALYSIS_CONFIG["gf_exp_values"]
    system_types = ANALYSIS_CONFIG["system_types"]
    num_messages = ANALYSIS_CONFIG["num_messages"]
    vec_len = ANALYSIS_CONFIG["vec_len"]
    message_subset_size = ANALYSIS_CONFIG["message_subset_size"]
    
    # Generate all combinations of gf_exp and system_type
    for gf_exp in gf_exp_values:
        for system_type in system_types:
            parameter_sets.append({
                "system_type": system_type,
                "gf_exp": gf_exp,
                "vec_len": vec_len,
                "num_messages": num_messages,
                "test_type": None,
                "tag_pos": None,
                "num_tags": None,
                "num_validation_messages": None
            })
    
    return parameter_sets


def analyze_single_combination(params):
    """
    Analyze a single parameter combination.
    
    Args:
        params: Dictionary with test parameters
        
    Returns:
        Dictionary with analysis results
    """
    # Create the system with specified parameters
    if params["system_type"] in ["RSID", "RMID"]:
        system = create_id_system(params["system_type"], {
            "gf_exp": params["gf_exp"], 
            "tag_pos": [2]
        })
    else:  # SHA1ID, SHA256ID
        system = create_id_system(params["system_type"], {
            "gf_exp": params["gf_exp"]
        })
    
    # Use evaluate_system for a single system
    results = IdMetrics.evaluate_system(
        system=system,
        vec_len=params["vec_len"],
        num_messages=params["num_messages"]
    )
    
    # Return all results from evaluate_system
    return results


def run_gf_exp_analysis_with_checkpointing():
    """Run the GF exponent analysis with checkpointing."""
    
    print("=" * 60)
    print("IDENTIFICATION SYSTEMS - GF_EXP INFLUENCE (WITH CHECKPOINTING)")
    print("=" * 60)
    
    # Create checkpoint manager
    output_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="gf_exp_influence",
        save_interval=1  # Save after each parameter combination
    )
    
    # Generate parameter combinations
    parameter_sets = create_parameter_sets()
    
    # Add comprehensive metadata about the analysis
    metadata = {
        "description": "Analysis of GF exponent influence on reliability and execution time",
        "gf_exp_values": ANALYSIS_CONFIG["gf_exp_values"],
        "system_types": ANALYSIS_CONFIG["system_types"],
        "num_messages": ANALYSIS_CONFIG["num_messages"],
        "vec_len": ANALYSIS_CONFIG["vec_len"],
        "message_subset_size": ANALYSIS_CONFIG["message_subset_size"],
        "total_combinations": len(parameter_sets)
    }
    
    # Initialize analysis (will resume from checkpoint if available)
    remaining_params = checkpoint.initialize_analysis(
        parameter_sets=parameter_sets,
        metadata=metadata
    )
    
    print(f"Processing {len(remaining_params)} remaining parameter combinations...")
    
    # Process each parameter combination
    for i, params in enumerate(remaining_params):
        test_desc = f"{params['system_type']}: gf_exp={params['gf_exp']}"
        print(f"\n[{i+1}/{len(remaining_params)}] Testing: {test_desc}")
        
        try:
            # Run the analysis
            results = analyze_single_combination(params)
            
            # Save result to checkpoint
            checkpoint.add_result(params, results)
            
            # Print results
            fp_rate = results["false_positive_rate"]
            exec_time = results["avg_execution_time_ms"]
            false_positives = results["false_positives"]
            
            print(f"   Results: FP={false_positives}, FP_rate={fp_rate:.6f}, Time={exec_time:.2f}ms")
            
            # Print progress
            completion = checkpoint.get_completion_percentage()
            print(f"   Progress: {completion:.1f}% complete")
            
        except Exception as e:
            print(f"   Error processing {params}: {e}")
            continue
    
    # Finalize the analysis
    checkpoint.finalize_analysis()
    
    # Get results DataFrame
    results_df = checkpoint.get_results_dataframe()
    
    print(f"\nAnalysis complete! Results saved to CSV with {len(results_df)} rows.")
    print(f"CSV file: {checkpoint.csv_file}")
    
    return results_df


def print_detailed_results(results_df):
    """Print detailed results similar to the original script format."""
    
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    # Group results by system type
    for system_type in results_df['system_type'].unique():
        subset = results_df[results_df['system_type'] == system_type]
        
        print(f"\n{system_type}:")
        print("-" * 40)
        
        for _, row in subset.iterrows():
            gf_exp = row['gf_exp']
            fp_rate = row['false_positive_rate']
            exec_time = row['avg_execution_time_ms']
            false_positives = row['false_positives']
            
            print(f"  gf_exp={gf_exp}: fp_rate={fp_rate:.6f}, "
                  f"exec_time={exec_time:.3f}ms, false_positives={false_positives}")


def create_analysis_summary(results_df):
    """Create a summary of the analysis results."""
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    total_combinations = len(results_df)
    num_systems = results_df['system_type'].nunique()
    num_gf_exp = results_df['gf_exp'].nunique()
    
    print(f"Total parameter combinations tested: {total_combinations}")
    print(f"Number of system types: {num_systems}")
    print(f"Number of GF exponents: {num_gf_exp}")
    
    print(f"\nSystem types tested: {sorted(results_df['system_type'].unique())}")
    print(f"GF exponents tested: {sorted(results_df['gf_exp'].unique())}")
    
    # Overall statistics
    print(f"\nOverall false positive rate range: {results_df['false_positive_rate'].min():.6f} - {results_df['false_positive_rate'].max():.6f}")
    print(f"Overall execution time range: {results_df['avg_execution_time_ms'].min():.3f} - {results_df['avg_execution_time_ms'].max():.3f} ms")
    
    # Find best and worst performers
    fastest_idx = results_df['avg_execution_time_ms'].idxmin()
    slowest_idx = results_df['avg_execution_time_ms'].idxmax()
    
    fastest = results_df.loc[fastest_idx]
    slowest = results_df.loc[slowest_idx]
    
    print(f"\nFastest combination:")
    print(f"  {fastest['system_type']}, gf_exp={fastest['gf_exp']}: {fastest['avg_execution_time_ms']:.3f} ms")
    
    print(f"\nSlowest combination:")
    print(f"  {slowest['system_type']}, gf_exp={slowest['gf_exp']}: {slowest['avg_execution_time_ms']:.3f} ms")


if __name__ == "__main__":
    # Run the analysis with checkpointing
    results_df = run_gf_exp_analysis_with_checkpointing()
    
    # Print detailed results (original script format)
    print_detailed_results(results_df)
    
    # Create analysis summary
    create_analysis_summary(results_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to CSV: analyses/gf_exp_influence/checkpoints/gf_exp_influence_results.csv")
