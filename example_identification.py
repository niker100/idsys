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


def compare_systems(
    systems: Dict[str, IdSystem],
    message_set: List[Any],
    code_lengths: List[int],
    num_trials: int = 100
) -> None:
    """
    Compare multiple identification systems across different metrics.
    
    Args:
        systems: Dictionary mapping system names to IdSystem instances
        message_set: Set of messages to use for testing
        code_lengths: List of code lengths to test
        num_trials: Number of trials for each metric calculation
    """
    # Setup visualization
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0:2])  # Reliability (main metric)
    ax2 = fig.add_subplot(gs[0, 2])     # False positive rate
    ax3 = fig.add_subplot(gs[1, 0])     # Collision probability
    ax4 = fig.add_subplot(gs[1, 1])     # Code rate
    ax5 = fig.add_subplot(gs[1, 2])     # Trade-off plot
    ax_table = fig.add_subplot(gs[2, :])  # Summary table
    ax_table.axis('off')
    
    fig.suptitle("Identification System Comparison", fontsize=16)
    
    # Colors and markers for different systems
    colors = sns.color_palette("muted", n_colors=len(systems))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    
    print("\nComparing identification systems...")
    comparison_data = {}
    
    # Collect data for all metrics at once to minimize redundant calculations
    for i, (name, system) in enumerate(systems.items()):
        print(f"Testing system: {name}")
        reliabilities = []
        fp_rates = []
        fn_rates = []
        collision_probs = []
        code_rates = []
        
        for code_length in code_lengths:
            # Set parameters for both encoder and decoder
            system.encoder.set_parameters({"code_length": code_length})
            system.decoder.set_parameters({"code_length": code_length})
            
            # Calculate all metrics
            reliability = IdMetrics.reliability(system, message_set, num_trials)
            error_rates = IdMetrics.error_rates(system, message_set, num_trials)
            collision_prob = IdMetrics.worst_case_collision_probability(
                system, message_set, sample_size=min(10, len(message_set)), num_trials=10
            )
            efficiency = IdMetrics.efficiency(system)
            
            # Store results
            reliabilities.append(reliability)
            fp_rates.append(error_rates["false_positive_rate"])
            fn_rates.append(error_rates["false_negative_rate"])
            collision_probs.append(collision_prob)
            code_rates.append(efficiency["code_rate"])
        
        # Store all data for this system
        comparison_data[name] = {
            "reliabilities": reliabilities,
            "fp_rates": fp_rates,
            "fn_rates": fn_rates,
            "collision_probs": collision_probs,
            "code_rates": code_rates
        }
        
        # Plot data for this system
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Reliability plot (main metric)
        ax1.plot(code_lengths, reliabilities, marker=marker, linestyle='-',
                color=color, linewidth=2, label=name)
        
        # FP rate plot
        ax2.plot(code_lengths, fp_rates, marker=marker, linestyle='-',
                color=color, linewidth=2, label=name)
        
        # Collision probability
        ax3.plot(code_lengths, collision_probs, marker=marker, linestyle='-',
                color=color, linewidth=2, label=name)
        
        # Code rate
        ax4.plot(code_lengths, code_rates, marker=marker, linestyle='-',
                color=color, linewidth=2, label=name)
        
        # Trade-off plot (reliability vs fp rate)
        ax5.scatter(reliabilities, fp_rates, s=80, c=[color], marker=marker, label=name, alpha=0.7)
        
    # Customize plots
    ax1.set_xlabel('Code Length (bits)')
    ax1.set_ylabel('Reliability')
    ax1.set_title('System Reliability vs Code Length', fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    ax2.set_xlabel('Code Length (bits)')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_title('False Positive Rate', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Code Length (bits)')
    ax3.set_ylabel('Collision Probability')
    ax3.set_title('Collision Probability', fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Code Length (bits)')
    ax4.set_ylabel('Code Rate')
    ax4.set_title('Code Rate (Efficiency)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5.set_xlabel('Reliability')
    ax5.set_ylabel('False Positive Rate')
    ax5.set_title('Reliability vs FP Rate Trade-off', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    
    # Generate summary data for table
    system_names = list(systems.keys())
    data_rows = []
    metrics = ["Avg. Reliability", "Avg. FP Rate", "Avg. Collision Prob.", "Avg. Code Rate"]
    
    for name in system_names:
        data = comparison_data[name]
        avg_rel = np.mean(data["reliabilities"])
        avg_fp = np.mean(data["fp_rates"])
        avg_coll = np.mean(data["collision_probs"])
        avg_rate = np.mean(data["code_rates"])
        data_rows.append([f"{avg_rel:.4f}", f"{avg_fp:.4f}", f"{avg_coll:.4f}", f"{avg_rate:.4f}"])
    
    # Create table
    table = ax_table.table(
        cellText=data_rows,
        rowLabels=system_names,
        colLabels=metrics,
        loc='center',
        cellLoc='center',
        colWidths=[0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        elif key[1] == -1:  # First column (system names)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9E1F2')
        else:
            cell.set_facecolor('#E9EDF4')
    
    # Add insights based on data
    optimal_lengths = {}
    for name, data in comparison_data.items():
        # Find optimal code length where reliability >= 0.95 and FP rate < 0.1
        optimal_idx = None
        for i, (rel, fp) in enumerate(zip(data["reliabilities"], data["fp_rates"])):
            if rel >= 0.95 and fp < 0.1:
                optimal_idx = i
                break
        if optimal_idx is not None:
            optimal_lengths[name] = code_lengths[optimal_idx]
    
    if optimal_lengths:
        optimal_text = "Optimal code lengths (reliability ≥ 0.95, FP rate < 0.1):\n"
        for name, length in optimal_lengths.items():
            optimal_text += f"• {name}: {length} bits\n"
        fig.text(0.02, 0.02, optimal_text, fontsize=10, va='bottom', ha='left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('system_comparison.png', dpi=300)
    print("Comparison analysis completed and saved as 'system_comparison.png'")


def explore_parameter_effects(
    system: IdSystem,
    message_set: List[Any],
    param_name: str,
    param_values: List[Any],
    num_trials: int = 100,
    set_on_decoder: bool = True
) -> None:
    """
    Explore how a parameter affects different metrics of the system.
    
    Args:
        system: The identification system to evaluate
        message_set: Set of messages to use for testing
        param_name: Name of the parameter to vary
        param_values: List of values to test for the parameter
        num_trials: Number of trials for each metric calculation
        set_on_decoder: If True, set the parameter on both encoder and decoder
    """
    # Setup visualization
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Reliability
    ax2 = fig.add_subplot(gs[0, 1])  # Error rates
    ax3 = fig.add_subplot(gs[1, 0])  # Collision probability
    ax4 = fig.add_subplot(gs[1, 1])  # Trade-off
    ax_heatmap = fig.add_subplot(gs[2, :])  # Impact heatmap
    
    fig.suptitle(f"Effect of {param_name} on System Performance", fontsize=16)
    
    print(f"\nAnalyzing effect of {param_name} on system performance...")
    
    # Store metrics
    reliabilities = []
    false_positives = []
    false_negatives = []
    collision_probs = []
    
    # Calculate all metrics for each parameter value
    for value in param_values:
        # Set parameters on both encoder and decoder or just the encoder
        system.encoder.set_parameters({param_name: value})
        if set_on_decoder:
            system.decoder.set_parameters({param_name: value})
        
        # Calculate metrics
        reliability = IdMetrics.reliability(system, message_set, num_trials)
        error_rates = IdMetrics.error_rates(system, message_set, num_trials)
        collision_probs = IdMetrics.worst_case_collision_probability(
            system, message_set, sample_size=min(10, len(message_set)), num_trials=10
        )
        
        # Store results
        reliabilities.append(reliability)
        false_positives.append(error_rates["false_positive_rate"])
        false_negatives.append(error_rates["false_negative_rate"])
        print(f"Value {value}: Reliability = {reliability:.4f}, FP Rate = {error_rates['false_positive_rate']:.4f}")
    
    # Plot results with improved formatting
    
    # Reliability plot
    sns.lineplot(x=param_values, y=reliabilities, ax=ax1, marker='o', linewidth=2, color='#4472C4')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Reliability')
    ax1.set_title(f'Reliability vs {param_name}', fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Error rates plot
    ax2.plot(param_values, false_positives, 'r-o', linewidth=2, label='False Positives')
    ax2.plot(param_values, false_negatives, 'g-s', linewidth=2, label='False Negatives')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Error Rate')
    ax2.set_title(f'Error Rates vs {param_name}', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Collision probability plot
    sns.lineplot(x=param_values, y=collision_probs, ax=ax3, marker='o', linewidth=2, color='#ED7D31')
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Collision Probability')
    ax3.set_title(f'Collision Probability vs {param_name}', fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    
    # Custom trade-off plot
    scatter = ax4.scatter(reliabilities, false_positives, c=param_values, 
                         cmap='viridis', s=100, alpha=0.7)
    # Add parameter value labels
    for i, value in enumerate(param_values):
        ax4.annotate(
            f"{value}",
            (reliabilities[i], false_positives[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    ax4.set_xlabel('Reliability')
    ax4.set_ylabel('False Positive Rate')
    ax4.set_title('Reliability vs FP Rate Trade-off', fontweight='bold')
    ax4.set_xlim(0, 1.05)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label(param_name)
    
    # Create heatmap of parameter impact
    # Normalize data for better visualization
    reliabilities_array = np.array(reliabilities)
    false_positives_array = np.array(false_positives)
    collision_probs_array = np.array(collision_probs)
    
    norm_reliability = (reliabilities_array - np.min(reliabilities_array)) / (np.max(reliabilities_array) - np.min(reliabilities_array) + 1e-10)
    norm_fp = (false_positives_array - np.min(false_positives_array)) / (np.max(false_positives_array) - np.min(false_positives_array) + 1e-10)
    norm_coll = (collision_probs_array - np.min(collision_probs_array)) / (np.max(collision_probs_array) - np.min(collision_probs_array) + 1e-10)
    
    # Higher values for reliability are better, but lower values for FP and collision are better
    impact_scores = norm_reliability - norm_fp - norm_coll
    
    # Create a heatmap with parameter values and their impact scores
    heatmap_data = np.array([impact_scores])
    sns.heatmap(heatmap_data, cmap="RdYlGn", ax=ax_heatmap, 
                xticklabels=[str(v) for v in param_values], 
                yticklabels=["Impact Score"],
                cbar_kws={'label': 'Overall Impact'})
    ax_heatmap.set_title(f'Impact of {param_name} on System Performance', fontweight='bold')
    
    # Add best value annotation
    best_idx = np.argmax(impact_scores)
    best_value = param_values[best_idx]
    best_reliability = reliabilities[best_idx]
    best_fp = false_positives[best_idx]
    
    note_text = (f"Best {param_name} value: {best_value}\n"
                f"Reliability: {best_reliability:.4f}\n"
                f"FP Rate: {best_fp:.4f}")
    
    fig.text(0.02, 0.02, note_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'parameter_effects_{param_name}.png', dpi=300)
    print(f"Parameter analysis completed and saved as 'parameter_effects_{param_name}.png'")


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
    
    # Compare some preset system configurations
    test_systems = {
        "Sys1: 32/4/8": {"nsize": 32, "nsym": 4, "code_length": 8},
        "Sys2: 32/8/16": {"nsize": 32, "nsym": 8, "code_length": 16},
        "Sys3: 64/16/12": {"nsize": 64, "nsym": 16, "code_length": 12},
        "Sys4: 96/24/20": {"nsize": 96, "nsym": 24, "code_length": 20}
    }
    
    results = []
    
    for name, params in test_systems.items():
        test_sys = create_id_system("paper_tagging", params)
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
    num_messages = 1000
    string_messages = generate_string_messages(num_messages, length=10)

    # Create identification systems for testing
    systems = {
        "RS-32/8/8": create_id_system("paper_tagging", {
            "nsize": 32,   # total codeword length (data + ECC)
            "nsym": 8,     # number of ECC symbols
            "code_length": 8  # length of tag sequence to extract
        }),
        "RS-32/8/16": create_id_system("paper_tagging", {
            "nsize": 32,   # total codeword length
            "nsym": 8,     # number of ECC symbols
            "code_length": 16  # length of tag sequence
        }),
        "RS-64/16/8": create_id_system("paper_tagging", {
            "nsize": 64,   # total codeword length
            "nsym": 16,    # number of ECC symbols
            "code_length": 8  # length of tag sequence
        })
    }

    # Basic correctness test
    is_valid = test_system_correctness(systems["RS-32/8/8"], string_messages)
    
    if not is_valid:
        print("⚠ Warning: Basic correctness test failed! Results may be unreliable.")
    
    # Compare system performance
    print("\nStep 1: Comparing system configurations...")
    code_lengths = [i for i in range(2, 25, 1)]
    compare_systems(systems, string_messages, code_lengths, num_trials=1000)
    
    # Parameter effect analysis
    print("\nStep 2: Analyzing parameter effects...")
    
    # Analyze effect of ECC symbols (nsym)
    print("Testing ECC symbol count effect...")
    rs_system = create_id_system("paper_tagging", {"nsize": 32, "nsym": 4, "code_length": 8})
    nsym_values = [i for i in range(2, 17, 1)]
    explore_parameter_effects(rs_system, string_messages, "nsym", nsym_values, num_trials=1000)
    
    # Analyze effect of code length
    print("Testing code length effect...")
    rs_system = create_id_system("paper_tagging", {"nsize": 32, "nsym": 8, "code_length": 8})
    code_lengths = [i for i in range(2, 25, 1)]
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