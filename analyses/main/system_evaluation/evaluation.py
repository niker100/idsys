#!/usr/bin/env python3
"""
Comprehensive benchmarking script for identification systems.

This script evaluates identification systems across multiple dimensions:
- System types: RSID, RS2ID, RMID, SHA1ID, SHA256ID
- Galois field exponents: 8, 16, 32, 64
- Vector lengths: 2^2 to 2^16
- Number of validation messages
- Number of tags (for multi-tag systems)
- Message patterns: random, sparse, low_entropy

Features:
- Automatic checkpointing for resume capability
- Progress tracking
- Memory efficiency by disabling PDF generation
- Parallel processing
"""

import sys
import os
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from idsys import IdMetrics, create_id_system, create_checkpoint_manager

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Analysis configuration - centralized to avoid duplication
ANALYSIS_CONFIG = {
    "system_types": ["RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"],
    "gf_exponents": [8, 16, 32, 64],
    "vector_lengths": [2**i for i in range(3, 17)],  # 8 to 65536
    "validation_messages": [2**i for i in range(1, 14)],  # 2^1 to 2^10
    "message_patterns": ["random", "sparse", "low_entropy"],
    "tag_counts": {
        "RSID": [1, 2, 3],
        "RMID": [1, 2, 3],
        "RS2ID": [1],
        "SHA1ID": [1],
        "SHA256ID": [1]
    },
    "num_messages": 10000000,  # Default number of messages for false positive testing
    "calculate_pdfs": False,  # Disable PDF calculation for memory efficiency
    "rm_orders": {
        "RMID": [1]  # Different Reed-Muller orders
    }
}

# Cached vector lengths for different system types
CACHED_VECTOR_LENGTHS = {
    "RSID": [2**i for i in range(3, 17)],    # Up to 65536
    "RS2ID": [2**i for i in range(3, 17)],   # Up to 65536
    "RMID": [2**i for i in range(3, 17)],    # Up to 65536
    "SHA1ID": [2**i for i in range(3, 17)],  # Up to 65536
    "SHA256ID": [2**i for i in range(3, 17)] # Up to 65536
}

def create_parameter_sets():
    """Generate all parameter combinations to test."""
    parameter_sets = []
    
    # Create two main test types:
    # 1. Execution Time Test (focusing on system performance)
    # 2. False Positive Rate Test (focusing on system reliability)
    
    # 1. EXECUTION TIME TESTS
    for system_type in ANALYSIS_CONFIG["system_types"]:
        # Use cached vector lengths for execution time tests
        vector_lengths = CACHED_VECTOR_LENGTHS.get(system_type, ANALYSIS_CONFIG["vector_lengths"])
        
        for gf_exp in ANALYSIS_CONFIG["gf_exponents"]:
            # Skip incompatible combinations
            if system_type == "RS2ID" and gf_exp > 32:
                continue
            if system_type == "RMID" and gf_exp > 32:
                continue
            
            # For each tag count compatible with system
            for tag_count in ANALYSIS_CONFIG["tag_counts"].get(system_type, [1]):
                # Generate tag positions
                if tag_count == 1:
                    tag_positions = [2]
                else:
                    tag_positions = list(range(2, 2 + tag_count))
                
                # For RM-based systems, test different orders
                rm_orders = ANALYSIS_CONFIG["rm_orders"].get(system_type, [None])
                for rm_order in rm_orders:
                    # Skip if incompatible (high order with high GF_EXP)
                    if system_type == "RMID" and rm_order > 1 and gf_exp > 32:
                        continue
                
                    for vec_len in vector_lengths:
                        # Skip very large tests for high gf_exp
                        if gf_exp >= 32 and vec_len > 2**14:
                            continue
                        
                        # Add execution time test (use fewer messages)
                        parameter_sets.append({
                            "test_type": "execution_time",
                            "system_type": system_type,
                            "gf_exp": gf_exp,
                            "vec_len": vec_len,
                            "tag_pos": tag_positions,
                            "num_tags": tag_count,
                            "num_validation_messages": 1,
                            "num_messages": min(1000000, max(100000, 1000000 // (vec_len // 100 + 1))),
                            "message_pattern": "random",
                            "rm_order": rm_order
                        })
    
    # 2. FALSE POSITIVE RATE TESTS
    for system_type in ANALYSIS_CONFIG["system_types"]:
        for gf_exp in [8, 16, 32, 64]:  # Limit to smaller GF sizes for FP rate tests

            if system_type == "RS2ID" and gf_exp > 32:
                continue
            if system_type == "RMID" and gf_exp > 32:
                continue
        
            # For each tag count compatible with system
            for tag_count in ANALYSIS_CONFIG["tag_counts"].get(system_type, [1]):
                # Generate tag positions
                if tag_count == 1:
                    tag_positions = [2]
                else:
                    tag_positions = list(range(2, 2 + tag_count))
                
                # For validation messages
                for num_validation_messages in ANALYSIS_CONFIG["validation_messages"]:
                    # For message patterns
                    for message_pattern in ANALYSIS_CONFIG["message_patterns"]:
                        # Keep vector length moderate for reliability tests
                        vec_len = 16
                        
                        # Use fewer messages for multi-tag and multi-validation tests
                        adjusted_messages = ANALYSIS_CONFIG["num_messages"]
                        # if tag_count > 1:
                        #     adjusted_messages //= tag_count
                        # if num_validation_messages > 10:
                        #     adjusted_messages //= (num_validation_messages // 10)
                        
                        # Add the parameter set
                        parameter_sets.append({
                            "test_type": "false_positive_rate",
                            "system_type": system_type,
                            "gf_exp": gf_exp,
                            "vec_len": vec_len,
                            "tag_pos": tag_positions,
                            "num_tags": tag_count,
                            "num_validation_messages": num_validation_messages,
                            "num_messages": adjusted_messages,
                            "message_pattern": message_pattern,
                            "rm_order": 1 if system_type == "RMID" else None
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
    print(f"Processing: {params}")
    
    # Create system parameters dictionary
    system_params = {
        "gf_exp": params["gf_exp"],
        "tag_pos": params["tag_pos"]
    }
    
    # Add RM order for RMID system
    if params["system_type"] == "RMID" and params["rm_order"] is not None:
        system_params["rm_order"] = params["rm_order"]
    
    # Create the system with specified parameters
    system = create_id_system(params["system_type"], system_params)
    
    # Run the evaluation
    start_time = time.time()
    results = IdMetrics.evaluate_system(
        system=system,
        vec_len=params["vec_len"],
        num_messages=params["num_messages"],
        num_validation_messages=params["num_validation_messages"],
        message_pattern=params["message_pattern"],
        calculate_pdfs=ANALYSIS_CONFIG["calculate_pdfs"]
    )
    elapsed_time = time.time() - start_time
    
    # Add analysis metadata
    results["total_analysis_time"] = elapsed_time
    results["analysis_timestamp"] = time.time()
    
    # Print a brief summary
    print(f"  Results - FP Rate: {results['false_positive_rate']:.6f}, "
          f"Avg Time: {results['avg_execution_time_ms']:.3f} ms, "
          f"Total Analysis Time: {elapsed_time:.1f}s")
    
    return results


def run_analysis_with_checkpointing():
    """Run the multi-parameter analysis with checkpointing."""
    
    print("=" * 80)
    print("COMPREHENSIVE IDENTIFICATION SYSTEMS BENCHMARK")
    print("=" * 80)
    
    # Create checkpoint manager
    output_dir = SCRIPT_DIR / "checkpoints"
    checkpoint = create_checkpoint_manager(
        output_dir=output_dir,
        analysis_name="multi_parameter_benchmark",
        save_interval=5  # Save after every 5 tests
    )
    
    # Generate parameter combinations
    parameter_sets = create_parameter_sets()
    
    print(f"Generated {len(parameter_sets)} parameter combinations")
    print(f"- Execution time tests: {sum(1 for p in parameter_sets if p['test_type'] == 'execution_time')}")
    print(f"- False positive rate tests: {sum(1 for p in parameter_sets if p['test_type'] == 'false_positive_rate')}")
    
    # Add comprehensive metadata about the analysis
    metadata = {
        "description": "Comprehensive benchmark of identification systems",
        "system_types": ANALYSIS_CONFIG["system_types"],
        "gf_exponents": ANALYSIS_CONFIG["gf_exponents"],
        "vector_lengths": ANALYSIS_CONFIG["vector_lengths"],
        "validation_messages": ANALYSIS_CONFIG["validation_messages"],
        "message_patterns": ANALYSIS_CONFIG["message_patterns"],
        "tag_counts": ANALYSIS_CONFIG["tag_counts"],
        "total_combinations": len(parameter_sets),
        "calculate_pdfs": ANALYSIS_CONFIG["calculate_pdfs"]
    }
    
    # Initialize analysis (will resume from checkpoint if available)
    remaining_params = checkpoint.initialize_analysis(
        parameter_sets=parameter_sets,
        metadata=metadata
    )
    
    print(f"\nProcessing {len(remaining_params)} remaining parameter combinations...")
    
    # Sort remaining parameters to optimize execution (group similar tests together)
    remaining_params.sort(key=lambda p: (
        p["test_type"], 
        p["system_type"], 
        p["gf_exp"], 
        p["vec_len"]
    ))
    
    # Process each parameter combination
    for i, params in enumerate(remaining_params):
        print(f"\n[{i+1}/{len(remaining_params)}] ", end="")
        
        try:
            # Run the analysis
            results = analyze_single_combination(params)
            
            # Save result to checkpoint
            checkpoint.add_result(params, results)
            
            # Print progress
            completion = checkpoint.get_completion_percentage()
            print(f"   Progress: {completion:.1f}% complete")
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Finalize the analysis
    checkpoint.finalize_analysis()
    
    # Get results DataFrame
    results_df = checkpoint.get_results_dataframe()
    
    print(f"\nAnalysis complete! Results saved to CSV with {len(results_df)} rows.")
    print(f"CSV file: {checkpoint.csv_file}")
    
    return results_df


def generate_summary_plots(results_df):
    """Generate summary plots from the benchmark results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogFormatter, LogLocator
        import seaborn as sns
        
        print("\nGenerating summary plots...")
        plots_dir = SCRIPT_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("talk")
        
        # 1. EXECUTION TIME PLOTS
        # -----------------------
        exec_time_df = results_df[results_df['test_type'] == 'execution_time'].copy()
        
        # Plot 1: Execution time vs vector length for different systems (GF_EXP=8)
        plt.figure(figsize=(12, 8))
        
        for system in ANALYSIS_CONFIG["system_types"]:
            system_data = exec_time_df[(exec_time_df['system_type'] == system) & 
                                      (exec_time_df['gf_exp'] == 8) & 
                                      (exec_time_df['num_tags'] == 1)]
            if not system_data.empty:
                plt.plot(system_data['vec_len'], system_data['avg_execution_time_ms'], 
                        marker='o', linewidth=2, label=system)
        
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.xlabel('Vector Length')
        plt.ylabel('Avg Execution Time (ms)')
        plt.title('Execution Time vs Vector Length (GF_EXP=8)')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(plots_dir / "execution_time_vs_veclen.png", dpi=300)
        
        # Plot 2: Execution time vs GF_EXP for different systems (vec_len=16)
        plt.figure(figsize=(12, 8))
        
        for system in ANALYSIS_CONFIG["system_types"]:
            system_data = exec_time_df[(exec_time_df['system_type'] == system) & 
                                      (exec_time_df['vec_len'] == 16) & 
                                      (exec_time_df['num_tags'] == 1)]
            if len(system_data) > 1:  # Need at least 2 points to plot a line
                plt.plot(system_data['gf_exp'], system_data['avg_execution_time_ms'], 
                        marker='o', linewidth=2, label=system)
        
        plt.xlabel('GF Exponent')
        plt.ylabel('Avg Execution Time (ms)')
        plt.title('Execution Time vs GF Exponent (vec_len=16)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "execution_time_vs_gfexp.png", dpi=300)
        
        # Plot 3: Execution time vs tag count for multi-tag systems
        plt.figure(figsize=(12, 8))
        
        for system in ["RSID", "RMID"]:
            system_data = exec_time_df[(exec_time_df['system_type'] == system) & 
                                      (exec_time_df['gf_exp'] == 8) & 
                                      (exec_time_df['vec_len'] == 16)]
            
            if not system_data.empty:
                # Group by tag count and calculate mean
                grouped = system_data.groupby('num_tags')['avg_execution_time_ms'].mean().reset_index()
                if len(grouped) > 1:  # Need at least 2 points to plot a line
                    plt.plot(grouped['num_tags'], grouped['avg_execution_time_ms'], 
                            marker='o', linewidth=2, label=system)
        
        plt.xlabel('Number of Tags')
        plt.ylabel('Avg Execution Time (ms)')
        plt.title('Execution Time vs Tag Count (GF_EXP=8, vec_len=16)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "execution_time_vs_tagcount.png", dpi=300)
        
        # 2. FALSE POSITIVE RATE PLOTS
        # ---------------------------
        fp_df = results_df[results_df['test_type'] == 'false_positive_rate'].copy()
        
        # Plot 4: FP rate vs validation messages for different systems
        plt.figure(figsize=(12, 8))
        
        for system in ANALYSIS_CONFIG["system_types"]:
            system_data = fp_df[(fp_df['system_type'] == system) & 
                              (fp_df['gf_exp'] == 8) & 
                              (fp_df['message_pattern'] == 'random') & 
                              (fp_df['num_tags'] == 1)]
            
            if not system_data.empty:
                # Group by validation messages and calculate mean
                grouped = system_data.groupby('num_validation_messages')['false_positive_rate'].mean().reset_index()
                if len(grouped) > 1:  # Need at least 2 points
                    plt.plot(grouped['num_validation_messages'], grouped['false_positive_rate'], 
                            marker='o', linewidth=2, label=system)
        
        plt.xlabel('Number of Validation Messages')
        plt.ylabel('False Positive Rate')
        plt.title('FP Rate vs Number of Validation Messages (GF_EXP=8, random pattern)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fprate_vs_validation.png", dpi=300)
        
        # Plot 5: FP rate vs tag count for multi-tag systems
        plt.figure(figsize=(12, 8))
        
        for system in ["RSID", "RMID"]:
            system_data = fp_df[(fp_df['system_type'] == system) & 
                              (fp_df['gf_exp'] == 8) & 
                              (fp_df['message_pattern'] == 'random') &
                              (fp_df['num_validation_messages'] == 1)]
            
            if not system_data.empty:
                # Group by tag count and calculate mean
                grouped = system_data.groupby('num_tags')['false_positive_rate'].mean().reset_index()
                if len(grouped) > 1:  # Need at least 2 points
                    plt.plot(grouped['num_tags'], grouped['false_positive_rate'], 
                            marker='o', linewidth=2, label=system)
        
        plt.xlabel('Number of Tags')
        plt.ylabel('False Positive Rate')
        plt.title('FP Rate vs Tag Count (GF_EXP=8, random pattern)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fprate_vs_tagcount.png", dpi=300)
        
        # Plot 6: FP rate vs message pattern for different systems
        plt.figure(figsize=(14, 8))
        
        message_patterns = fp_df['message_pattern'].unique()
        system_names = []
        fp_rates = []
        patterns = []
        
        for system in ANALYSIS_CONFIG["system_types"]:
            for pattern in message_patterns:
                system_data = fp_df[(fp_df['system_type'] == system) & 
                                  (fp_df['gf_exp'] == 8) & 
                                  (fp_df['message_pattern'] == pattern) & 
                                  (fp_df['num_validation_messages'] == 1) &
                                  (fp_df['num_tags'] == 1)]
                
                if not system_data.empty:
                    mean_fp = system_data['false_positive_rate'].mean()
                    system_names.append(system)
                    fp_rates.append(mean_fp)
                    patterns.append(pattern)
        
        # Create a dataframe for easier plotting
        pattern_df = pd.DataFrame({
            'System': system_names,
            'FP Rate': fp_rates,
            'Pattern': patterns
        })
        
        # Plot as grouped bar chart
        sns.barplot(x='System', y='FP Rate', hue='Pattern', data=pattern_df)
        plt.title('FP Rate by System and Message Pattern (GF_EXP=8)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "fprate_by_pattern.png", dpi=300)
        
        print(f"Plots saved to {plots_dir}")
        
    except ImportError:
        print("Warning: Could not generate plots. Please install matplotlib and seaborn.")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    # Determine number of processes for analysis
    num_cpus = mp.cpu_count()
    num_processes = max(1, num_cpus - 1)  # Leave one CPU for system
    
    print(f"Starting benchmark with {num_processes} worker processes...")
    
    # Run the analysis with checkpointing
    results_df = run_analysis_with_checkpointing()
    
    # Generate summary plots
    # generate_summary_plots(results_df)
    
    # Display some example commands for analyzing results
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Results saved to CSV. Here are some commands for further analysis:")
    print("\n# Open the CSV file in your browser:")
    print(f"$BROWSER {SCRIPT_DIR}/checkpoints/multi_parameter_benchmark_results.csv")
    print("\n# To analyze the data later, run:")
    print("python -c \"import pandas as pd; df = pd.read_csv('analyses/multi_parameter_benchmark/checkpoints/multi_parameter_benchmark_results.csv'); print(df.head())\"")
    print("\n# To generate additional plots, you can use the results DataFrame")
    print("="*80)