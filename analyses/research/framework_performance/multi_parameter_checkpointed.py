#!/usr/bin/env python3
"""
Enhanced version of multi_parameter analysis with checkpointing support.

This script plots execution time vs vector length for multiple gf_exp values and systems.
Each analysis generates all gf_exp curves for one system across multiple vector lengths.

Features:
- Automatic periodic saving to CSV
- Resume capability after crashes
- Progress tracking
- Backup creation
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from idsys import IdMetrics, create_id_system, create_checkpoint_manager


# Analysis configuration - centralized to avoid duplication
ANALYSIS_CONFIG = {
    "vec_lengths": [2**i for i in range(1, 16)],
    "gf_exp_values": [8, 16, 32, 64],
    "system_types": [
        ("RSID", lambda gf_exp: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]})),
        ("RMID", lambda gf_exp: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2]})),
        ("RS2ID", lambda gf_exp: create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": [2], "tag_pos_in": 2})),
        ("SHA1ID", lambda gf_exp: create_id_system("SHA1ID", {"gf_exp": gf_exp})),
        ("SHA256ID", lambda gf_exp: create_id_system("SHA256ID", {"gf_exp": gf_exp})),
    ],
    "num_messages": 100,
    "message_subset_size": 10
}


def create_parameter_sets():
    """Generate all parameter combinations to test."""
    parameter_sets = []
    
    # Use centralized configuration
    vec_lengths = ANALYSIS_CONFIG["vec_lengths"]
    gf_exp_values = ANALYSIS_CONFIG["gf_exp_values"]
    system_types = ANALYSIS_CONFIG["system_types"]
    num_messages = ANALYSIS_CONFIG["num_messages"]
    
    # Consistent parameter set format for CSV comparability
    for system_name, sys_factory in system_types:
        for gf_exp in gf_exp_values:
            for vec_len in vec_lengths:
                parameter_sets.append({
                    "system_type": system_name,
                    "gf_exp": gf_exp,
                    "vec_len": vec_len,
                    "num_messages": num_messages,
                    "test_type": None,
                    "tag_pos": None,
                    "num_tags": None,
                    "num_validation_messages": None
                })
    
    return parameter_sets


def get_system_factory(system_type):
    """Get the system factory function for a given system type."""
    system_factories = {
        "RSID": lambda gf_exp: create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        "RMID": lambda gf_exp: create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        "RS2ID": lambda gf_exp: create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": [2], "tag_pos_in": 2}),
        "SHA1ID": lambda gf_exp: create_id_system("SHA1ID", {"gf_exp": gf_exp}),
        "SHA256ID": lambda gf_exp: create_id_system("SHA256ID", {"gf_exp": gf_exp}),
    }
    return system_factories.get(system_type)


def analyze_single_combination(params):
    """
    Analyze a single parameter combination.
    
    Args:
        params: Dictionary with test parameters
        
    Returns:
        Dictionary with analysis results
    """
    # Get the system factory function and create the system
    system_factory = get_system_factory(params["system_type"])
    if system_factory is None:
        raise ValueError(f"Unknown system type: {params['system_type']}")
    
    system = system_factory(params["gf_exp"])
    
    # Run the evaluation
    results = IdMetrics.evaluate_system(
        system=system,
        vec_len=params["vec_len"],
        num_messages=params["num_messages"]
    )
    
    # Return all results from evaluate_system
    return results


def run_multi_parameter_analysis_with_checkpointing():
    """Run the multi-parameter analysis with checkpointing."""
    
    print("=" * 70)
    print("EXECUTION TIME VS VECTOR LENGTH FOR MULTIPLE GF_EXP AND SYSTEMS")
    print("(WITH CHECKPOINTING)")
    print("=" * 70)
    
    # Create checkpoint manager
    output_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="multi_parameter_performance",
        save_interval=5  # Save every 5 parameter combinations
    )
    
    # Generate parameter combinations
    parameter_sets = create_parameter_sets()
    
    # Add comprehensive metadata about the analysis
    metadata = {
        "description": "Execution time vs vector length analysis for multiple GF exponents and system types",
        "vec_lengths": ANALYSIS_CONFIG["vec_lengths"],
        "gf_exp_values": ANALYSIS_CONFIG["gf_exp_values"],
        "system_types": [name for name, _ in ANALYSIS_CONFIG["system_types"]],
        "num_messages": ANALYSIS_CONFIG["num_messages"],
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
        test_desc = f"{params['system_type']}: gf_exp={params['gf_exp']}, vec_len={params['vec_len']}"
        print(f"\n[{i+1}/{len(remaining_params)}] Testing: {test_desc}")
        
        try:
            # Run the analysis
            results = analyze_single_combination(params)
            
            # Save result to checkpoint (params are already serializable now)
            checkpoint.add_result(params, results)
            
            # Print results
            exec_time = results["avg_execution_time_ms"]
            print(f"   Execution time: {exec_time:.3f} ms")
            
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
    """Print detailed results summary."""
    
    print("\n" + "="*70)
    print("DETAILED RESULTS SUMMARY")
    print("="*70)
    
    # Group results by system type
    for system_type in results_df['system_type'].unique():
        system_data = results_df[results_df['system_type'] == system_type]
        
        print(f"\n{system_type}:")
        print("-" * 40)
        
        # Group by gf_exp
        for gf_exp in sorted(system_data['gf_exp'].unique()):
            gf_data = system_data[system_data['gf_exp'] == gf_exp]
            
            min_time = gf_data['avg_execution_time_ms'].min()
            max_time = gf_data['avg_execution_time_ms'].max()
            avg_time = gf_data['avg_execution_time_ms'].mean()
            
            print(f"  GF_EXP={gf_exp}: {len(gf_data)} measurements")
            print(f"    Execution time range: {min_time:.3f} - {max_time:.3f} ms (avg: {avg_time:.3f} ms)")
            
            # Show vec_len range
            min_vec = gf_data['vec_len'].min()
            max_vec = gf_data['vec_len'].max()
            print(f"    Vector length range: {min_vec} - {max_vec}")


def create_analysis_summary(results_df):
    """Create a summary of the analysis results."""
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    total_combinations = len(results_df)
    num_systems = results_df['system_type'].nunique()
    num_gf_exp = results_df['gf_exp'].nunique()
    num_vec_lengths = results_df['vec_len'].nunique()
    
    print(f"Total parameter combinations tested: {total_combinations}")
    print(f"Number of system types: {num_systems}")
    print(f"Number of GF exponents: {num_gf_exp}")
    print(f"Number of vector lengths: {num_vec_lengths}")
    
    print(f"\nSystem types tested: {sorted(results_df['system_type'].unique())}")
    print(f"GF exponents tested: {sorted(results_df['gf_exp'].unique())}")
    print(f"Vector length range: {results_df['vec_len'].min()} - {results_df['vec_len'].max()}")
    
    # Overall execution time statistics
    overall_min = results_df['avg_execution_time_ms'].min()
    overall_max = results_df['avg_execution_time_ms'].max()
    overall_avg = results_df['avg_execution_time_ms'].mean()
    overall_std = results_df['avg_execution_time_ms'].std()
    
    print(f"\nOverall execution time statistics:")
    print(f"  Minimum: {overall_min:.3f} ms")
    print(f"  Maximum: {overall_max:.3f} ms")
    print(f"  Average: {overall_avg:.3f} ms")
    print(f"  Std Dev: {overall_std:.3f} ms")
    
    # Find fastest and slowest combinations
    fastest_idx = results_df['avg_execution_time_ms'].idxmin()
    slowest_idx = results_df['avg_execution_time_ms'].idxmax()
    
    fastest = results_df.loc[fastest_idx]
    slowest = results_df.loc[slowest_idx]
    
    print(f"\nFastest combination:")
    print(f"  {fastest['system_type']}, GF_EXP={fastest['gf_exp']}, vec_len={fastest['vec_len']}: {fastest['avg_execution_time_ms']:.3f} ms")
    
    print(f"\nSlowest combination:")
    print(f"  {slowest['system_type']}, GF_EXP={slowest['gf_exp']}, vec_len={slowest['vec_len']}: {slowest['avg_execution_time_ms']:.3f} ms")


if __name__ == "__main__":
    # Run the analysis with checkpointing
    results_df = run_multi_parameter_analysis_with_checkpointing()
    
    # Print detailed results
    print_detailed_results(results_df)
    
    # Create analysis summary
    create_analysis_summary(results_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to CSV: analyses/framework_performance/checkpoints/multi_parameter_performance_results.csv")
    print("\nTo create plots from this data, you can load the CSV file and use matplotlib.")
    print("The CSV contains all combinations of system_name, gf_exp, vec_len with their execution times.")
