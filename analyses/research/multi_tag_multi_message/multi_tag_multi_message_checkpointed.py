#!/usr/bin/env python3
"""
Enhanced version of multi_tag_multi_message analysis with checkpointing support.

This script evaluates the false positive rates for different combinations of:
- Number of tags (tag positions)  
- Number of validation messages
- Their combinations

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
    "tag_positions": [[2], [2, 3], [2, 3, 4]],
    "nums_validation_messages": [10, 50, 100],
    "vec_len": 16,
    "num_messages": 100000,
    "gf_exp": 8,
    "system_type": "RSID"
}


def create_parameter_sets():
    """Generate all parameter combinations to test."""
    parameter_sets = []
    
    # Use centralized configuration
    tag_positions = ANALYSIS_CONFIG["tag_positions"]
    nums_validation_messages = ANALYSIS_CONFIG["nums_validation_messages"]
    vec_len = ANALYSIS_CONFIG["vec_len"]
    num_messages = ANALYSIS_CONFIG["num_messages"]
    
    # Test 1: Multiple validation messages with single tag
    for num_validation_messages in nums_validation_messages:
        parameter_sets.append({
            "system_type": ANALYSIS_CONFIG["system_type"],
            "gf_exp": ANALYSIS_CONFIG["gf_exp"],
            "vec_len": vec_len,
            "num_messages": num_messages,
            "test_type": "multi_validation_single_tag",
            "tag_pos": [2],
            "num_tags": 1,
            "num_validation_messages": num_validation_messages
        })
    
    # Test 2: Multiple tags with single validation message
    for tag_pos in tag_positions:
        parameter_sets.append({
            "system_type": ANALYSIS_CONFIG["system_type"],
            "gf_exp": ANALYSIS_CONFIG["gf_exp"],
            "vec_len": vec_len,
            "num_messages": num_messages,
            "test_type": "multi_tag_single_validation",
            "tag_pos": tag_pos,
            "num_tags": len(tag_pos),
            "num_validation_messages": 1
        })
    
    # Test 3: Combination of multiple tags and multiple validation messages
    for num_validation_messages in nums_validation_messages:
        parameter_sets.append({
            "system_type": ANALYSIS_CONFIG["system_type"],
            "gf_exp": ANALYSIS_CONFIG["gf_exp"],
            "vec_len": vec_len,
            "num_messages": num_messages,
            "test_type": "multi_tag_multi_validation",
            "tag_pos": [2, 3],
            "num_tags": 2,
            "num_validation_messages": num_validation_messages
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
    # Create the system with specified tag positions
    system = create_id_system(ANALYSIS_CONFIG["system_type"], {
        "gf_exp": ANALYSIS_CONFIG["gf_exp"], 
        "tag_pos": params["tag_pos"]
    })
    
    # Run the evaluation
    results = IdMetrics.evaluate_system(
        system=system,
        vec_len=params["vec_len"],
        num_messages=params["num_messages"],
        num_validation_messages=params["num_validation_messages"]
    )
    # Only return results from evaluate_system (do not add theoretical_fp_rate or num_tags)
    return results


def run_multi_tag_analysis_with_checkpointing():
    """Run the multi-tag multi-message analysis with checkpointing."""
    
    print("=" * 60)
    print("MULTI-TAG MULTI-MESSAGE ANALYSIS (WITH CHECKPOINTING)")
    print("=" * 60)
    
    # Create checkpoint manager
    output_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="multi_tag_multi_message",
        save_interval=1  # Save after each test combination
    )
    
    # Generate parameter combinations
    parameter_sets = create_parameter_sets()
    
    # Add comprehensive metadata about the analysis
    metadata = {
        "description": "Analysis of false positive rates for multiple tags and validation messages",
        "system_type": ANALYSIS_CONFIG["system_type"],
        "gf_exp": ANALYSIS_CONFIG["gf_exp"],
        "tag_positions": ANALYSIS_CONFIG["tag_positions"],
        "nums_validation_messages": ANALYSIS_CONFIG["nums_validation_messages"],
        "vec_len": ANALYSIS_CONFIG["vec_len"],
        "num_messages": ANALYSIS_CONFIG["num_messages"],
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
        test_desc = f"{params['test_type']}: tags={params['tag_pos']}, val_msgs={params['num_validation_messages']}"
        print(f"\n[{i+1}/{len(remaining_params)}] Testing: {test_desc}")
        
        try:
            # Run the analysis
            results = analyze_single_combination(params)
            
            # Save result to checkpoint
            checkpoint.add_result(params, results)
            
            # Print results (only those from evaluate_system)
            fp_rate = results.get("false_positive_rate", None)
            print(f"   fp_rate: {fp_rate}")
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
    
    # Group results by test type
    for test_type in results_df['test_type'].unique():
        subset = results_df[results_df['test_type'] == test_type]
        
        print(f"\n{test_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        for _, row in subset.iterrows():
            fp_rate = row.get('false_positive_rate', None)
            num_val_msgs = row.get('num_validation_messages', None)
            tag_pos = row.get('tag_pos', None)
            if test_type == "multi_validation_single_tag":
                print(f"  num_validation_messages={num_val_msgs}: fp_rate={fp_rate}")
            elif test_type == "multi_tag_single_validation":
                print(f"  tags={tag_pos}: fp_rate={fp_rate}")
            else:  # multi_tag_multi_validation
                print(f"  tags={tag_pos}, num_validation_messages={num_val_msgs}: fp_rate={fp_rate}")


if __name__ == "__main__":
    # Run the analysis with checkpointing
    results_df = run_multi_tag_analysis_with_checkpointing()
    
    # Print detailed results (original script format)
    print_detailed_results(results_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to CSV: analyses/multi_tag_multi_message/checkpoints/multi_tag_multi_message_results.csv")
