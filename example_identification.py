#!/usr/bin/env python3
"""
Example script demonstrating the use of the identification system framework.

This script shows how to:
1. Create different types of identification systems
2. Evaluate system performance using various metrics
3. Visualize the results with different plots
4. Compare multiple identification systems
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from typing import List, Dict, Any
import os
import sys
from matplotlib.gridspec import GridSpec

from framework import (
    create_id_system, IdSystem,
    generate_numeric_messages, generate_string_messages,
    IdMetrics
)
from framework.utils import compare_systems, explore_parameter_effects


def test_system_correctness(system: IdSystem, message_set: List[Any], num_samples: int = 5):
    """
    Test basic correctness of an identification system.
    
    Args:
        system: The identification system to test
        message_set: Set of messages to test with
        num_samples: Number of messages to sample for testing
    """
    if num_samples > len(message_set):
        num_samples = len(message_set)
        
    sample_messages = message_set[:num_samples]
    
    print("\nValidating system correctness...")
    
    results = {
        "tests_performed": 0,
        "passed": 0,
        "failed": 0,
        "false_positives": 0,
        "false_negatives": 0
    }
    
    for message in sample_messages:
        # Encode
        codeword = system.send(message)
        
        # Decode with same message (should be True)
        result_same = system.receive(codeword, message)
        
        # Decode with different message (should be False)
        different_idx = (sample_messages.index(message) + 1) % len(sample_messages)
        different_message = sample_messages[different_idx]
        result_diff = system.receive(codeword, different_message)
        
        # Update results
        results["tests_performed"] += 1
        
        if result_same and not result_diff:
            results["passed"] += 1
        else:
            results["failed"] += 1
            if not result_same:
                results["false_negatives"] += 1
            if result_diff:
                results["false_positives"] += 1
    
    # Print summary
    print(f"System validation summary:")
    print(f"✓ Tests passed: {results['passed']}/{results['tests_performed']} ({results['passed']/results['tests_performed']*100:.1f}%)")
    if results["failed"] > 0:
        print(f"✗ Tests failed: {results['failed']}/{results['tests_performed']}")
        print(f"  - False positives: {results['false_positives']}")
        print(f"  - False negatives: {results['false_negatives']}")
    
    return results["passed"] == results["tests_performed"]


def create_parameter_optimization_dashboard(systems: Dict[str, IdSystem], message_set: List[Any]):
    """
    Create a comprehensive dashboard showing optimal parameter combinations.
    
    Args:
        systems: Dictionary mapping system names to IdSystem instances
        message_set: Set of messages to use for testing
    """
    # Setup visualization
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])  # nsym vs code_length for reliability
    ax2 = fig.add_subplot(gs[0, 1])  # nsym vs code_length for FP rate
    ax3 = fig.add_subplot(gs[1, 0])  # nsize vs code_length for reliability
    ax4 = fig.add_subplot(gs[1, 1])  # 3D parameter space visualization
    
    fig.suptitle("Parameter Optimization Dashboard", fontsize=16)
    
    # Define parameter ranges
    code_lengths = [2, 4, 6, 8, 12, 16, 20, 24]
    nsym_values = [2, 4, 6, 8, 10, 12, 14, 16]
    nsize_values = [32, 48, 64, 96, 128]
    
    # Prepare data structures for heatmaps
    rel_data_nsym = np.zeros((len(nsym_values), len(code_lengths)))
    fp_data_nsym = np.zeros((len(nsym_values), len(code_lengths)))
    rel_data_nsize = np.zeros((len(nsize_values), len(code_lengths)))
    
    # Base system for testing
    base_system = create_id_system("paper_tagging", {"nsize": 32, "nsym": 8, "code_length": 8})
    
    print("\nCreating parameter optimization dashboard...")
    print("This may take some time. Testing parameter combinations...")
    
    # Test nsym vs code_length combinations
    for i, nsym in enumerate(nsym_values):
        for j, code_length in enumerate(code_lengths):
            base_system.encoder.set_parameters({"nsym": nsym, "code_length": code_length})
            base_system.decoder.set_parameters({"nsym": nsym, "code_length": code_length})
            
            reliability = IdMetrics.reliability(base_system, message_set, num_trials=50)
            error_rates = IdMetrics.error_rates(base_system, message_set, num_trials=50)
            
            rel_data_nsym[i, j] = reliability
            fp_data_nsym[i, j] = error_rates["false_positive_rate"]
    
    # Test nsize vs code_length combinations
    for i, nsize in enumerate(nsize_values):
        for j, code_length in enumerate(code_lengths):
            base_system.encoder.set_parameters({"nsize": nsize, "code_length": code_length})
            base_system.decoder.set_parameters({"nsize": nsize, "code_length": code_length})
            
            reliability = IdMetrics.reliability(base_system, message_set, num_trials=50)
            rel_data_nsize[i, j] = reliability
    
    # Create heatmaps
    sns.heatmap(rel_data_nsym, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax1)
    ax1.set_title("Reliability: nsym vs code_length", fontweight='bold')
    ax1.set_xlabel("Code Length")
    ax1.set_ylabel("ECC Symbols (nsym)")
    
    sns.heatmap(fp_data_nsym, annot=True, fmt=".2f", cmap="YlOrRd_r", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax2)
    ax2.set_title("False Positive Rate: nsym vs code_length", fontweight='bold')
    ax2.set_xlabel("Code Length")
    ax2.set_ylabel("ECC Symbols (nsym)")
    
    sns.heatmap(rel_data_nsize, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=code_lengths, yticklabels=nsize_values, ax=ax3)
    ax3.set_title("Reliability: nsize vs code_length", fontweight='bold')
    ax3.set_xlabel("Code Length")
    ax3.set_ylabel("Codeword Length (nsize)")
    
    # 3D visualization with system comparison
    from mpl_toolkits.mplot3d import Axes3D
    ax4.remove()
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    
    results = []
    
    for name, test_sys in systems.items():
        params = test_sys.encoder.parameters
        reliability = IdMetrics.reliability(test_sys, message_set, num_trials=50)
        error_rates = IdMetrics.error_rates(test_sys, message_set, num_trials=50)
        results.append((name, params["nsize"], params["nsym"], params["code_length"], 
                        reliability, error_rates["false_positive_rate"]))
    
    # Plot 3D points
    colors = sns.color_palette("bright", n_colors=len(results))
    
    for i, (name, nsize, nsym, code_length, rel, fp) in enumerate(results):
        ax4.scatter(code_length, nsym, rel, color=colors[i], s=100, label=name, alpha=0.7)
        ax4.text(code_length, nsym, rel + 0.05, name, color=colors[i])
    
    ax4.set_xlabel('Code Length')
    ax4.set_ylabel('ECC Symbols (nsym)')
    ax4.set_zlabel('Reliability')
    ax4.set_title('3D Optimization Space', fontweight='bold')
    ax4.view_init(30, 45)
    ax4.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Identify optimal configurations
    best_rel_idx = np.unravel_index(np.argmax(rel_data_nsym), rel_data_nsym.shape)
    best_fp_idx = np.unravel_index(np.argmin(fp_data_nsym), fp_data_nsym.shape)
    
    best_rel_nsym = nsym_values[best_rel_idx[0]]
    best_rel_code = code_lengths[best_rel_idx[1]]
    best_fp_nsym = nsym_values[best_fp_idx[0]]
    best_fp_code = code_lengths[best_fp_idx[1]]
    
    insight_text = (
        f"Optimal Configurations:\n\n"
        f"Best for Reliability:\n"
        f"• nsym = {best_rel_nsym}, code_length = {best_rel_code}\n"
        f"• Reliability: {rel_data_nsym[best_rel_idx]:.4f}\n\n"
        f"Best for Low FP Rate:\n"
        f"• nsym = {best_fp_nsym}, code_length = {best_fp_code}\n"
        f"• FP Rate: {fp_data_nsym[best_fp_idx]:.4f}"
    )
    
    fig.text(0.02, 0.02, insight_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('parameter_optimization_dashboard.png', dpi=300)
    print("Parameter optimization dashboard created and saved as 'parameter_optimization_dashboard.png'")


def main():
    """Main function demonstrating the identification framework."""
    print("Identification System Analysis Framework")
    print("-" * 50)
    
    # Create output directory for figures
    os.makedirs("output", exist_ok=True)
    os.chdir("output")
    
    # Generate message sets for testing
    num_messages = 256
    string_messages = generate_string_messages(num_messages, length=8)

    # Create identification systems for testing
    systems = {
        "RS-64/4/8": create_id_system("paper_tagging", {
            "nsize": 64,   # total codeword length (data + ECC)
            "nsym": 4,     # number of ECC symbols
            "code_length": 8  # length of tag sequence to extract
        }),
        "RS-64/8/8": create_id_system("paper_tagging", {
            "nsize": 64,   # total codeword length
            "nsym": 8,     # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
        "RS-64/16/8": create_id_system("paper_tagging", {
            "nsize": 64,    # total codeword length
            "nsym": 16,      # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
        "RS-64/32/8": create_id_system("paper_tagging", {
            "nsize": 64,   # total codeword length
            "nsym": 32,     # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
    }
    
    # Compare system performance
    print("\nStep 1: Comparing system configurations...")
    code_lengths = [i for i in range(2, 64, 1)]
    compare_systems(systems, string_messages, code_lengths, num_trials=1000)
    
    # Parameter effect analysis
    print("\nStep 2: Analyzing parameter effects...")
    
    # Analyze effect of ECC symbols (nsym)
    print("Testing ECC symbol count effect...")
    rs_system = create_id_system("paper_tagging", {"nsize": 64, "nsym": 8, "code_length": 16})
    nsym_values = [i for i in range(2, 56, 1)]
    explore_parameter_effects(rs_system, string_messages, "nsym", nsym_values, num_trials=1000)
    
    # Analyze effect of code length
    print("Testing code length effect...")
    rs_system = create_id_system("paper_tagging", {"nsize": 64, "nsym": 8, "code_length": 8})
    code_lengths = [i for i in range(2, 64, 1)]
    explore_parameter_effects(rs_system, string_messages, "code_length", code_lengths, num_trials=1000)
    
    # Create comprehensive parameter optimization dashboard
    print("\nStep 3: Building parameter optimization dashboard...")
    create_parameter_optimization_dashboard(systems, string_messages)
    
    print("\nAnalysis complete. All visualizations saved to the output directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")