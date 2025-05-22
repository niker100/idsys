#!/usr/bin/env python3
"""
System Comparison and Optimization Script.

This script demonstrates:
1. Performance analysis for different RS tagging configurations
2. Parameter optimization for maximum effective code rate with reliability > 0.95
3. 3D visualization of the optimization results across alphabet sizes
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Tuple

from framework import (
    create_id_system,
    IdMetrics,
    utils
)

# Constants
OUTPUT_DIR = "output"
ALPHABET_SIZES = [2, 3, 4, 8, 16]  # Different alphabets to test
MAX_ENCODED_SIZE = 256  # message_length + nsym <= 256
RELIABILITY_THRESHOLD = 0.95
DEFAULT_TRIALS = 1000  # Monte Carlo trials


def analyze_code_length_effect(
    system_config: Dict[str, Any],
    messages: List[str],
    code_lengths: List[int] = None,
    num_trials: int = DEFAULT_TRIALS
) -> Dict[str, List[float]]:
    """
    Analyze the effect of different code lengths on system performance.
    
    Args:
        system_config: Base configuration for the system
        messages: Test messages to use
        code_lengths: List of code lengths to test
        num_trials: Number of Monte Carlo trials for each measurement
        
    Returns:
        Dictionary with measurement results
    """
    if code_lengths is None:
        code_lengths = list(range(1, 17))
        
    print(f"\nAnalyzing effect of code length (nsym={system_config['nsym']}, msg_len={system_config['message_length']})...")
    
    results = {
        'code_lengths': code_lengths,
        'reliabilities': [],
        'fp_rates': [],
        'effective_code_rates': []
    }
    
    for code_length in code_lengths:
        # Create a new system with the current code length
        system_config['code_length'] = code_length
        system = create_id_system("paper_tagging", system_config)
        
        # Measure performance metrics
        reliability = IdMetrics.reliability(system, messages, num_trials)
        error_rates = IdMetrics.error_rates(system, messages, num_trials)
        efficiency = IdMetrics.efficiency(system)
        
        print(f"Code Length: {code_length}")
        print(f"  Reliability: {reliability:.4f}")
        print(f"  False Positive Rate: {error_rates['false_positive_rate']:.4f}")
        print(f"  Effective Code Rate: {efficiency['effective_code_rate']:.4f}")
        
        results['reliabilities'].append(reliability)
        results['fp_rates'].append(error_rates['false_positive_rate'])
        results['effective_code_rates'].append(efficiency['effective_code_rate'])
    
    # Create visualization
    utils.plot_performance_metrics_dual_scale(
        metric_values={
            'Reliability': results['reliabilities'],
            'False Positive Rate': results['fp_rates'],
            'Code Rate': results['effective_code_rates']
        },
        x_values=code_lengths,
        x_label='Code Length (symbols)',
        title=f'Effect of Code Length (nsym={system_config["nsym"]}, msg_len={system_config["message_length"]})',
        filename=os.path.join(OUTPUT_DIR, f'code_length_effect_nsym{system_config["nsym"]}.png')
    )
    
    return results


def explore_parameter_space(
    alphabet_size: int,
    code_length: int = 1,
    num_trials: int = DEFAULT_TRIALS
) -> Dict[str, Any]:
    """
    Explore the parameter space for a given alphabet size and code length.
    
    Args:
        alphabet_size: Alphabet size to test
        code_length: Code length to use (fixed)
        num_trials: Number of Monte Carlo trials for each configuration
        
    Returns:
        Dictionary with optimization results
    """
    print(f"\nExploring parameter space for alphabet size {alphabet_size}, code_length {code_length}...")
    
    # Initialize result arrays
    message_lengths = []
    nsym_values = []
    reliabilities = []
    effective_code_rates = []
    
    # Test combinations of message_length and nsym
    for message_length in range(16, MAX_ENCODED_SIZE - 8, 4):
        test_messages = utils.generate_test_messages(500, message_length, alphabet_size)
        for nsym in range(8, MAX_ENCODED_SIZE - message_length, 4):
            # Skip if total encoded size exceeds limit
            if message_length + nsym > MAX_ENCODED_SIZE:
                continue
                
            # Create system configuration
            system_config = {
                "message_length": message_length,
                "nsym": nsym,
                "code_length": code_length
            }
            
            system = create_id_system("paper_tagging", system_config)
            
            # Measure reliability and effective code rate
            reliability = IdMetrics.reliability(system, test_messages, num_trials)
            effective_code_rate = IdMetrics.efficiency(system)["effective_code_rate"]
            
            # Store results
            message_lengths.append(message_length)
            nsym_values.append(nsym)
            reliabilities.append(reliability)
            effective_code_rates.append(effective_code_rate)
            
            print(f"Config: msg_len={message_length}, nsym={nsym} → Reliability: {reliability:.4f}, Code Rate: {effective_code_rate:.4f}")
    
    # Create a dataset for visualization
    results = {
        'message_lengths': np.array(message_lengths),
        'nsym_values': np.array(nsym_values),
        'reliabilities': np.array(reliabilities),
        'effective_code_rates': np.array(effective_code_rates)
    }
    
    # Find optimal configuration
    valid_configs = np.where(results['reliabilities'] >= RELIABILITY_THRESHOLD)[0]
    
    if len(valid_configs) > 0:
        # Find the configuration with maximum effective code rate among valid configurations
        best_idx = valid_configs[np.argmax(np.array(effective_code_rates)[valid_configs])]
        
        optimal_config = {
            'message_length': message_lengths[best_idx],
            'nsym': nsym_values[best_idx],
            'code_length': code_length,
            'reliability': reliabilities[best_idx],
            'effective_code_rate': effective_code_rates[best_idx]
        }
        
        print(f"\nOptimal configuration found:")
        print(f"  Message Length: {optimal_config['message_length']}")
        print(f"  ECC Symbols: {optimal_config['nsym']}")
        print(f"  Code Length: {optimal_config['code_length']}")
        print(f"  Reliability: {optimal_config['reliability']:.4f}")
        print(f"  Effective Code Rate: {optimal_config['effective_code_rate']:.4f}")
    else:
        print(f"\nNo configuration meets the reliability threshold of {RELIABILITY_THRESHOLD}")
        optimal_config = None
    
    # Create visualization of parameter space
    plot_parameter_space(results, alphabet_size, code_length, optimal_config)
    
    return {
        'explored_space': results,
        'optimal_config': optimal_config
    }


def plot_parameter_space(
    results: Dict[str, np.ndarray],
    alphabet_size: int,
    code_length: int,
    optimal_config: Dict[str, Any] = None
) -> None:
    """
    Visualize the parameter space exploration results.
    
    Args:
        results: Dictionary containing exploration results
        alphabet_size: Alphabet size used
        code_length: Code length used
        optimal_config: Optimal configuration (if found)
    """
    # Filter only configurations meeting reliability threshold
    valid_mask = results['reliabilities'] >= RELIABILITY_THRESHOLD
    
    # Create scatter plot of valid configurations
    plt.figure(figsize=(10, 8))
    
    # Plot all explored points with transparency
    plt.scatter(
        results['message_lengths'], 
        results['nsym_values'],
        c='lightgray', 
        alpha=0.3,
        s=30
    )
    
    # Plot valid configurations with color mapped to effective code rate
    if np.any(valid_mask):
        scatter = plt.scatter(
            results['message_lengths'][valid_mask], 
            results['nsym_values'][valid_mask],
            c=results['effective_code_rates'][valid_mask], 
            cmap='viridis',
            s=80,
            alpha=0.8
        )
        
        plt.colorbar(scatter, label='Effective Code Rate')
        
        # Highlight optimal configuration if available
        if optimal_config:
            plt.scatter(
                optimal_config['message_length'], 
                optimal_config['nsym'],
                marker='*', 
                s=300, 
                color='red',
                edgecolor='black',
                label=f"Optimal: Rate={optimal_config['effective_code_rate']:.2f}"
            )
            plt.legend()
    
    plt.title(f'Parameter Space for Alphabet Size {alphabet_size}, Code Length {code_length}')
    plt.xlabel('Message Length (symbols)')
    plt.ylabel('ECC Symbols (nsym)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'parameter_space_a{alphabet_size}_c{code_length}.png'), dpi=300)
    plt.close()


def optimize_across_alphabets(
    alphabet_sizes: List[int] = ALPHABET_SIZES,
    code_length: int = 1,
    num_trials: int = DEFAULT_TRIALS
) -> Dict[int, Dict[str, Any]]:
    """
    Find optimal configurations across different alphabet sizes.
    
    Args:
        alphabet_sizes: List of alphabet sizes to test
        code_length: Code length to use (fixed)
        num_trials: Number of Monte Carlo trials for each configuration
        
    Returns:
        Dictionary mapping alphabet sizes to their optimal configurations
    """
    print(f"\nOptimizing configurations across alphabet sizes...")
    print(f"Code Length: {code_length}, Reliability Threshold: {RELIABILITY_THRESHOLD}")
    
    optimal_configs = {}
    
    for alphabet_size in alphabet_sizes:
        print(f"\n{'-'*50}")
        print(f"ALPHABET SIZE: {alphabet_size}")
        print(f"{'-'*50}")
        
        optimization_result = explore_parameter_space(
            alphabet_size=alphabet_size,
            code_length=code_length,
            num_trials=num_trials
        )
        
        optimal_configs[alphabet_size] = optimization_result['optimal_config']
    
    # Create visualization comparing optimal configurations
    plot_optimal_comparison(optimal_configs)
    plot_3d_optimization_comparison(optimal_configs)
    
    return optimal_configs


def plot_optimal_comparison(optimal_configs: Dict[int, Dict[str, Any]]) -> None:
    """
    Create a comparison plot of optimal configurations for different alphabet sizes.
    
    Args:
        optimal_configs: Dictionary mapping alphabet sizes to optimal configurations
    """
    alphabet_sizes = []
    effective_rates = []
    message_lengths = []
    nsym_values = []
    
    for alphabet_size, config in optimal_configs.items():
        if config:
            alphabet_sizes.append(alphabet_size)
            effective_rates.append(config['effective_code_rate'])
            message_lengths.append(config['message_length'])
            nsym_values.append(config['nsym'])
    
    if not alphabet_sizes:
        print("No valid configurations to plot.")
        return
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot effective code rate vs alphabet size
    ax1.plot(alphabet_sizes, effective_rates, 'o-', linewidth=2, markersize=10)
    ax1.set_xlabel('Alphabet Size')
    ax1.set_ylabel('Effective Code Rate')
    ax1.set_title('Optimal Effective Code Rate by Alphabet Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot message length and nsym vs alphabet size
    ax2.bar(np.array(alphabet_sizes) - 0.2, message_lengths, width=0.4, label='Message Length')
    ax2.bar(np.array(alphabet_sizes) + 0.2, nsym_values, width=0.4, label='ECC Symbols')
    ax2.set_xlabel('Alphabet Size')
    ax2.set_ylabel('Number of Symbols')
    ax2.set_title('Optimal Parameter Configuration by Alphabet Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_configuration_comparison.png'), dpi=300)
    plt.close()


def plot_3d_optimization_comparison(optimal_configs: Dict[int, Dict[str, Any]]) -> None:
    """
    Create a 3D visualization of the optimization results.
    
    Args:
        optimal_configs: Dictionary mapping alphabet sizes to optimal configurations
    """
    # Extract data for plotting
    alphabet_sizes = []
    message_lengths = []
    nsym_values = []
    effective_rates = []
    
    for alphabet_size, config in optimal_configs.items():
        if config:
            alphabet_sizes.append(alphabet_size)
            message_lengths.append(config['message_length'])
            nsym_values.append(config['nsym'])
            effective_rates.append(config['effective_code_rate'])
    
    if not alphabet_sizes:
        print("No valid configurations for 3D plot.")
        return
    
    # Create 3D plot with better styling
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with reasonable marker size
    marker_size = 100
    scatter = ax.scatter(
        alphabet_sizes, 
        message_lengths, 
        nsym_values,
        c=effective_rates, 
        s=marker_size,
        cmap='viridis',
        edgecolors='black',
        alpha=0.9,
        marker='o'
    )
    
    # Add vertical lines to ground plane for better spatial perception
    for i in range(len(alphabet_sizes)):
        ax.plot(
            [alphabet_sizes[i], alphabet_sizes[i]], 
            [message_lengths[i], message_lengths[i]], 
            [0, nsym_values[i]], 
            color='gray', 
            linestyle='--', 
            alpha=0.5,
            linewidth=1.0
        )
    
    # Connect points with a line to show trend
    ax.plot(
        alphabet_sizes,
        message_lengths,
        nsym_values,
        color='crimson',
        linestyle='-',
        linewidth=2,
        alpha=0.7,
        marker='o',
        markersize=6
    )
    
    # Label each point with alphabet size and effective rate
    for i, (a, m, n, r) in enumerate(zip(alphabet_sizes, message_lengths, nsym_values, effective_rates)):
        ax.text(
            a, 
            m, 
            n + 5,  # Offset text above point
            f'A{a}: {r:.3f}', 
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='bottom'
        )
    
    # Set labels and title with better formatting
    ax.set_xlabel('Alphabet Size', fontsize=12, labelpad=10)
    ax.set_ylabel('Message Length', fontsize=12, labelpad=10)
    ax.set_zlabel('ECC Symbols (nsym)', fontsize=12, labelpad=10)
    plt.title('Optimal Configurations Across Alphabet Sizes', fontsize=14, pad=20)
    
    # Improve axis formatting
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set sensible axis limits with padding
    x_padding = max(2, (max(alphabet_sizes) - min(alphabet_sizes)) * 0.2)
    ax.set_xlim(min(alphabet_sizes) - x_padding, max(alphabet_sizes) + x_padding)
    
    y_padding = (max(message_lengths) - min(message_lengths)) * 0.1
    ax.set_ylim(min(message_lengths) - y_padding, max(message_lengths) + y_padding)
    
    ax.set_zlim(0, max(nsym_values) * 1.2)  # Start from 0 with 20% padding on top
    
    # Add colorbar with better formatting
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, aspect=30)
    cbar.set_label('Effective Code Rate', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # Add a grid for better depth perception
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set an optimized view angle
    ax.view_init(elev=25, azim=30)
    
    # Add a light gray ground plane for better depth perception
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2)
    )
    
    zz = np.zeros(xx.shape)
    
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_3d_visualization.png'), dpi=300)
    plt.close()


def main():
    """Main function demonstrating the system comparison and optimization."""
    start_time = time.time()
    
    print("Identification System Comparison & Optimization")
    print("=" * 50)
    
    # Setup output directory and visualization style
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    utils.setup_visualization_style(OUTPUT_DIR)
    
    # Step 1: Find optimal configurations for different alphabet sizes
    print("\nStep 1: Finding optimal configurations for different alphabet sizes...")
    optimal_configs = optimize_across_alphabets(
        alphabet_sizes=ALPHABET_SIZES,
        code_length=1,
        num_trials=DEFAULT_TRIALS
    )
    
    # Step 2: Analyze code length effect for the optimal configuration of alphabet size 2
    if 2 in optimal_configs and optimal_configs[2]:
        print("\nStep 2: Analyzing code length effect for optimal configuration (alphabet size 2)...")
        opt_config = optimal_configs[2]
        system_config = {
            "message_length": opt_config["message_length"],
            "nsym": opt_config["nsym"],
            "code_length": 1  # Start with code_length=1
        }
        
        messages = utils.generate_test_messages(500, system_config["message_length"], 2)
        analyze_code_length_effect(system_config, messages, list(range(1, 17)))
    
    # Print summary of optimization results
    print("\nSummary of Optimal Configurations:")
    for alphabet_size, config in sorted(optimal_configs.items()):
        if config:
            print(f"Alphabet Size {alphabet_size}:")
            print(f"  Message Length: {config['message_length']}")
            print(f"  ECC Symbols: {config['nsym']}")
            print(f"  Code Length: {config['code_length']}")
            print(f"  Reliability: {config['reliability']:.4f}")
            print(f"  Effective Code Rate: {config['effective_code_rate']:.4f}")
        else:
            print(f"Alphabet Size {alphabet_size}: No valid configuration found")
    
    # Report execution time
    elapsed = time.time() - start_time
    print(f"\nAnalysis complete. All visualizations saved to {OUTPUT_DIR}.")
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()