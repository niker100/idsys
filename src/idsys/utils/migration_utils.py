#!/usr/bin/env python3
"""
Migration utility to convert existing analysis scripts to use checkpointing.

This script provides helper functions to easily add checkpointing to your existing
analysis workflows with minimal code changes.
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path


def convert_json_to_csv(json_file_path: str, output_csv_path: str = None):
    """
    Convert existing JSON results files to CSV format.
    
    Args:
        json_file_path: Path to the JSON file with results
        output_csv_path: Path for the output CSV file (optional)
    """
    json_path = Path(json_file_path)
    
    if output_csv_path is None:
        output_csv_path = json_path.with_suffix('.csv')
    
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract system results and metadata
        if 'system_results' in data:
            system_results = data['system_results']
            metadata = {k: v for k, v in data.items() if k != 'system_results'}
            
            # Convert to DataFrame format
            rows = []
            for system_name, results in system_results.items():
                # Assume all lists in results have the same length
                if not results:
                    continue
                
                first_key = list(results.keys())[0]
                num_rows = len(results[first_key])
                
                for i in range(num_rows):
                    row = {'system_type': system_name}
                    
                    # Add metadata as constant columns
                    for meta_key, meta_value in metadata.items():
                        if isinstance(meta_value, (int, float, str)):
                            row[meta_key] = meta_value
                    
                    # Add results for this row
                    for result_key, result_values in results.items():
                        row[result_key] = result_values[i]
                    
                    rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output_csv_path, index=False)
            
            print(f"Successfully converted {json_path} to {output_csv_path}")
            print(f"Created {len(df)} rows with columns: {list(df.columns)}")
            
            return df
            
    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return None


def quick_checkpoint_wrapper(analysis_function, 
                            parameter_sets, 
                            output_dir,
                            analysis_name,
                            save_interval=5):
    """
    Quick wrapper to add checkpointing to any analysis function.
    
    Args:
        analysis_function: Function that takes (params) and returns results dict
        parameter_sets: List of parameter dictionaries
        output_dir: Directory to save checkpoints
        analysis_name: Name for the analysis
        save_interval: Save every N iterations
    
    Example:
        def my_analysis(params):
            # Your existing analysis code
            return {"metric1": value1, "metric2": value2}
        
        results = quick_checkpoint_wrapper(
            my_analysis, 
            parameter_sets,
            "my_analysis_output",
            "my_analysis"
        )
    """
    # Import here to avoid circular imports
    from idsys import create_checkpoint_manager
    
    # Create checkpoint manager
    checkpoint = create_checkpoint_manager(output_dir, analysis_name, save_interval)
    
    # Initialize analysis
    remaining_params = checkpoint.initialize_analysis(parameter_sets)
    
    print(f"Processing {len(remaining_params)} parameter combinations...")
    
    # Process each parameter set
    for i, params in enumerate(remaining_params):
        print(f"[{i+1}/{len(remaining_params)}] Processing: {params}")
        
        try:
            # Run the analysis function
            results = analysis_function(params)
            
            # Save the result
            checkpoint.add_result(params, results)
            
            # Print progress
            completion = checkpoint.get_completion_percentage()
            print(f"   Progress: {completion:.1f}% complete")
            
        except Exception as e:
            print(f"Error processing {params}: {e}")
            continue
    
    # Finalize
    checkpoint.finalize_analysis()
    
    return checkpoint.get_results_dataframe()


# Example: Convert your existing analysis to use checkpointing with minimal changes
def convert_gf_exp_analysis_example():
    """
    Example showing how to convert your existing gf_exp analysis.
    """
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src import IdMetrics, create_id_system
    
    def analyze_single_combination(params):
        """
        Your existing analysis logic converted to process a single parameter combination.
        """
        gf_exp = params['gf_exp']
        system_type = params['system_type']
        num_messages = params['num_messages']
        vec_len = params['vec_len']
        
        # Create system (this is your existing code)
        if system_type == "RSID":
            system = create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": 2})
        elif system_type == "RMID":
            system = create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": 2})
        elif system_type == "SHA1ID":
            system = create_id_system("SHA1ID", {"gf_exp": gf_exp})
        else:
            raise ValueError(f"Unknown system type: {system_type}")
        
        systems = {system_type: system}
        
        # Run evaluation (this is your existing code)
        metrics = IdMetrics.compare_systems(
            systems=systems,
            num_messages=num_messages,
            vec_len=vec_len,
            message_subset_size=0
        )
        
        # Extract results (this is your existing code, slightly modified)
        system_metrics = metrics[system_type]
        
        return {
            "false_positives": system_metrics["false_positives"],
            "avg_execution_time_ms": system_metrics["avg_execution_time_ms"],
            "false_positive_rate": system_metrics["false_positive_rate"]
        }
    
    # Generate parameter combinations (this is your existing code, reorganized)
    gf_exp_values = [8, 16, 32, 64]
    system_types = ["RSID", "RMID", "SHA1ID"]
    num_messages = 1000000
    vec_len = 125
    
    parameter_sets = []
    for gf_exp in gf_exp_values:
        for system_type in system_types:
            parameter_sets.append({
                "gf_exp": gf_exp,
                "system_type": system_type,
                "num_messages": num_messages,
                "vec_len": vec_len
            })
    
    # Run with checkpointing (this is the new part!)
    results_df = quick_checkpoint_wrapper(
        analysis_function=analyze_single_combination,
        parameter_sets=parameter_sets,
        output_dir="analyses/gf_exp_influence/checkpoints_simple",
        analysis_name="gf_exp_simple",
        save_interval=1  # Save after each combination
    )
    
    print(f"Analysis complete! {len(results_df)} results saved.")
    return results_df


if __name__ == "__main__":
    print("Migration Utility for Checkpointing")
    print("=" * 40)
    
    # Convert existing JSON files to CSV
    json_files = [
        "analyses/gf_exp_influence/system_results.json",
        "analyses/num_messages_influence/system_results.json"
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"\nConverting {json_file}...")
            convert_json_to_csv(json_file)
        else:
            print(f"File not found: {json_file}")
    
    # Run example analysis
    print("\nRunning example checkpointed analysis...")
    try:
        results = convert_gf_exp_analysis_example()
        print("Example completed successfully!")
    except Exception as e:
        print(f"Example failed: {e}")
