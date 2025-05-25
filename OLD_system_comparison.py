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
import psutil
import gc

from framework import (
    create_id_system,
    IdMetrics,
    utils_old
)

p = psutil.Process(os.getpid())
p.cpu_affinity([1])  # Pin to specific CPU core for performance consistency, especially important for systems with P and E cores as longer running tasks may be scheduled on E cores
p.nice(psutil.HIGH_PRIORITY_CLASS)  # Set high priority for the process

# Constants
OUTPUT_DIR = "output/system_comparison"
ALPHABET_SIZES = [2, 3, 4, 8, 16]  # Different alphabets to test
NSYM_CURVES = 4  # Number of different nsym curves to test
MAX_ENCODED_SIZE = 255  # message_length + nsym <= 255
MESSAGE_LENGTHS = [i for i in range(8, MAX_ENCODED_SIZE - 8, 8)]  # Message lengths to test

RELIABILITY_THRESHOLD = 0.95
DEFAULT_TRIALS = 5000  # Monte Carlo trials
VERBOSE = True  # Set to True for verbose output



def measure_computation_time_from_reliability(system, messages: List[str], num_trials: int = 100) -> Tuple[float, float]:
    """
    Measure computation time by timing the reliability function.
    
    Args:
        system: The identification system
        messages: List of test messages
        num_trials: Number of trials for timing measurement (same as reliability measurement)
        
    Returns:
        Tuple of (reliability, average_time_per_operation_ms)
    """
    # import gc
    
    # Disable garbage collection during timing for more accurate results
    # gc.disable()
    
    start_time = time.perf_counter()
    reliability = IdMetrics.reliability(system, messages, num_trials)
    end_time = time.perf_counter()
    
    # gc.enable()
    # gc.collect()  # Force garbage collection after timing
    
    # Calculate total time and average per operation
    total_time_ms = (end_time - start_time) * 1000
    
    # Each trial in reliability() does one encoding and one decoding operation
    # So total operations = 2 * num_trials
    average_time_per_operation_ms = total_time_ms / (2 * num_trials)
    
    return reliability, average_time_per_operation_ms


def explore_parameter_space(
    alphabet_size: int,
    code_length: int = 1,
    num_trials: int = 100,
    num_nsym_curves: int = 4
) -> Dict[str, Any]:
    """
    Explore the parameter space for a given alphabet size and code length.
    Tests multiple nsym values to create different curves.
    
    Args:
        alphabet_size: Alphabet size to test
        code_length: Code length to use (fixed)
        num_trials: Number of Monte Carlo trials for each configuration
        num_nsym_curves: Number of different nsym curves to test
        
    Returns:
        Dictionary with optimization results
    """
    print(f"\nExploring parameter space for alphabet size {alphabet_size}, code_length {code_length}...")
    print(f"Testing {num_nsym_curves} different nsym curves")
    
    # Calculate nsym fractions to test
    nsym_fractions = []
    for i in range(num_nsym_curves):
        fraction = 1.0 - (i / num_nsym_curves)  # 1.0, 0.75, 0.5, 0.25 for 4 curves
        nsym_fractions.append(fraction)
    
    print(f"Testing nsym fractions: {[f'{f:.2f}' for f in nsym_fractions]}")
    
    # Initialize result storage for all curves
    all_results = {}
    all_configs = []

    system_config = {
        "message_length": 1,  # Placeholder, will be set in the loop
        "nsym": 1,  # Placeholder, will be set in the loop
        "code_length": code_length
    }

    system = create_id_system("paper_tagging", system_config)
    
    # Test each nsym fraction
    for curve_idx, nsym_fraction in enumerate(nsym_fractions):
        print(f"\n--- Curve {curve_idx + 1}: {nsym_fraction:.2f} × max nsym ---")
        
        # Initialize result arrays for this curve
        message_lengths = []
        nsym_values = []
        reliabilities = []
        effective_code_rates = []
        total_computation_times = []
        computational_efficiencies = []
        
        # Test different message lengths with this nsym fraction
        for message_length in MESSAGE_LENGTHS:
            # Calculate nsym for this fraction
            max_nsym = MAX_ENCODED_SIZE - message_length
            nsym = max(1, int(max_nsym * nsym_fraction))  # Ensure at least 1
            
            # Skip if this would exceed the Reed-Solomon limit
            if message_length + nsym > MAX_ENCODED_SIZE:
                continue

            if VERBOSE: 
                print(f"Testing: msg_len={message_length}, nsym={nsym} ({nsym_fraction:.2f} × {max_nsym})")
            
            # Generate test messages
            test_messages = utils_old.generate_test_messages(100, message_length, alphabet_size)
            
            system.decoder.set_parameters({
                "message_length": message_length,
                "nsym": nsym,
                "code_length": code_length
            })
            system.encoder.set_parameters({
                "message_length": message_length,
                "nsym": nsym,
                "code_length": code_length
            })
            
            # Measure reliability and computation time together
            reliability, avg_time_per_operation = measure_computation_time_from_reliability(
                system, test_messages, num_trials
            )
            
            # Get effective code rate
            effective_code_rate = IdMetrics.efficiency(system)["effective_code_rate"]
            
            # Calculate computational efficiency (code rate per unit time)
            computational_efficiency = effective_code_rate / avg_time_per_operation * 1000  # per second
            
            if VERBOSE:
                print(f"  Reliability: {reliability:.4f}")
                print(f"  Code Rate: {effective_code_rate:.4f}")
                print(f"  Avg Time per Operation: {avg_time_per_operation:.4f} ms")
                print(f"  Computational Efficiency: {computational_efficiency:.2f}")
            
            # Store results for this curve
            message_lengths.append(message_length)
            nsym_values.append(nsym)
            reliabilities.append(reliability)
            effective_code_rates.append(effective_code_rate)
            total_computation_times.append(avg_time_per_operation)
            computational_efficiencies.append(computational_efficiency)
            
            # Store individual configuration for global optimization
            all_configs.append({
                'curve_idx': curve_idx,
                'nsym_fraction': nsym_fraction,
                'message_length': message_length,
                'nsym': nsym,
                'code_length': code_length,
                'reliability': reliability,
                'effective_code_rate': effective_code_rate,
                'avg_time_per_operation': avg_time_per_operation,
                'total_computation_time': avg_time_per_operation,  # Keep for compatibility
                'computational_efficiency': computational_efficiency
            })
        
        # Store results for this curve
        curve_results = {
            'message_lengths': np.array(message_lengths),
            'nsym_values': np.array(nsym_values),
            'reliabilities': np.array(reliabilities),
            'effective_code_rates': np.array(effective_code_rates),
            'total_computation_times': np.array(total_computation_times),
            'computational_efficiencies': np.array(computational_efficiencies),
            'nsym_fraction': nsym_fraction
        }
        
        all_results[f'curve_{curve_idx}'] = curve_results
    
    # Find optimal configurations across ALL curves
    valid_configs = [config for config in all_configs if config['reliability'] >= RELIABILITY_THRESHOLD]
    
    if valid_configs:
        # Find configuration with maximum effective code rate across all curves
        optimal_config = max(valid_configs, key=lambda x: x['effective_code_rate'])
        
        # Find configuration with maximum computational efficiency across all curves
        most_efficient_config = max(valid_configs, key=lambda x: x['computational_efficiency'])
        
        print(f"\nGLOBAL Optimal configuration (max effective code rate across all curves):")
        print(f"  Curve: {optimal_config['curve_idx'] + 1} (nsym fraction: {optimal_config['nsym_fraction']:.2f})")
        print(f"  Message Length: {optimal_config['message_length']}")
        print(f"  ECC Symbols: {optimal_config['nsym']}")
        print(f"  Code Length: {optimal_config['code_length']}")
        print(f"  Reliability: {optimal_config['reliability']:.4f}")
        print(f"  Effective Code Rate: {optimal_config['effective_code_rate']:.4f}")
        print(f"  Avg Time per Operation: {optimal_config['avg_time_per_operation']:.4f} ms")
        print(f"  Computational Efficiency: {optimal_config['computational_efficiency']:.2f}")
        
        print(f"\nGLOBAL Most efficient configuration (max computational efficiency across all curves):")
        print(f"  Curve: {most_efficient_config['curve_idx'] + 1} (nsym fraction: {most_efficient_config['nsym_fraction']:.2f})")
        print(f"  Message Length: {most_efficient_config['message_length']}")
        print(f"  ECC Symbols: {most_efficient_config['nsym']}")
        print(f"  Code Length: {most_efficient_config['code_length']}")
        print(f"  Reliability: {most_efficient_config['reliability']:.4f}")
        print(f"  Effective Code Rate: {most_efficient_config['effective_code_rate']:.4f}")
        print(f"  Avg Time per Operation: {most_efficient_config['avg_time_per_operation']:.4f} ms")
        print(f"  Computational Efficiency: {most_efficient_config['computational_efficiency']:.2f}")
        
    else:
        print(f"\nNo configuration meets the reliability threshold of {RELIABILITY_THRESHOLD}")
        optimal_config = None
        most_efficient_config = None
    
    # Create visualization with multiple curves
    plot_parameter_space_multi_curve(all_results, alphabet_size, code_length, optimal_config, most_efficient_config)
    
    return {
        'explored_space': all_results,
        'optimal_config': optimal_config,
        'most_efficient_config': most_efficient_config,
        'all_configs': all_configs
    }


def plot_parameter_space_multi_curve(
    all_results: Dict[str, Dict[str, np.ndarray]],
    alphabet_size: int,
    code_length: int,
    optimal_config: Dict[str, Any] = None,
    most_efficient_config: Dict[str, Any] = None
) -> None:
    """
    Visualize the parameter space exploration results with multiple nsym curves.
    
    Args:
        all_results: Dictionary containing results for all curves
        alphabet_size: Alphabet size used
        code_length: Code length used
        optimal_config: Global optimal configuration (if found)
        most_efficient_config: Global most efficient configuration (if found)
    """
    # Create a comprehensive visualization with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Parameter Space Analysis - Alphabet Size {alphabet_size} (Multiple nsym Curves)', fontsize=16)
    
    # Color map for different curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    
    # Plot each curve
    for i, (curve_name, results) in enumerate(all_results.items()):
        color = colors[i]
        nsym_fraction = results['nsym_fraction']
        label = f'nsym = {nsym_fraction:.2f}×max'
        
        # 1. Computational Efficiency vs Message Length
        ax1.plot(results['message_lengths'], results['computational_efficiencies'], 
                'o-', color=color, linewidth=2, alpha=0.8, label=label)
        
        # 2. Reliability vs Message Length
        ax2.plot(results['message_lengths'], results['reliabilities'], 
                'o-', color=color, linewidth=2, alpha=0.8, label=label)
        
        # 3. Computation Time vs Message Length
        ax3.plot(results['message_lengths'], results['total_computation_times'], 
                'o-', color=color, linewidth=2, alpha=0.8, label=label)
        
        # 4. Trade-off: Code Rate vs Computation Time
        ax4.scatter(results['effective_code_rates'], results['total_computation_times'], 
                   c=[color], s=40, alpha=0.7, label=label)
    
    # Add global optimal points
    if optimal_config:
        ax1.scatter(optimal_config['message_length'], optimal_config['computational_efficiency'], 
                   marker='*', s=300, color='red', label='Global Max Code Rate', zorder=10, edgecolors='black')
        ax2.scatter(optimal_config['message_length'], optimal_config['reliability'], 
                   marker='*', s=300, color='red', zorder=10, edgecolors='black')
        ax3.scatter(optimal_config['message_length'], optimal_config['total_computation_time'], 
                   marker='*', s=300, color='red', zorder=10, edgecolors='black')
        ax4.scatter(optimal_config['effective_code_rate'], optimal_config['total_computation_time'], 
                   marker='*', s=300, color='red', zorder=10, edgecolors='black')
    
    if most_efficient_config:
        ax1.scatter(most_efficient_config['message_length'], most_efficient_config['computational_efficiency'], 
                   marker='D', s=200, color='blue', label='Global Max Efficiency', zorder=10, edgecolors='black')
        ax2.scatter(most_efficient_config['message_length'], most_efficient_config['reliability'], 
                   marker='D', s=200, color='blue', zorder=10, edgecolors='black')
        ax3.scatter(most_efficient_config['message_length'], most_efficient_config['total_computation_time'], 
                   marker='D', s=200, color='blue', zorder=10, edgecolors='black')
        ax4.scatter(most_efficient_config['effective_code_rate'], most_efficient_config['total_computation_time'], 
                   marker='D', s=200, color='blue', zorder=10, edgecolors='black')
    
    # Configure subplots
    ax1.set_xlabel('Message Length')
    ax1.set_ylabel('Computational Efficiency (Code Rate / ms × 1000)')
    ax1.set_title('Computational Efficiency vs Message Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.axhline(y=RELIABILITY_THRESHOLD, color='red', linestyle='--', alpha=0.7, 
               label=f'Threshold ({RELIABILITY_THRESHOLD})')
    ax2.set_xlabel('Message Length')
    ax2.set_ylabel('Reliability')
    ax2.set_title('Reliability vs Message Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3.set_xlabel('Message Length')
    ax3.set_ylabel('Computation Time (ms)')
    ax3.set_title('Computation Time vs Message Length')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax4.set_xlabel('Effective Code Rate')
    ax4.set_ylabel('Total Computation Time (ms)')
    ax4.set_title('Trade-off: Code Rate vs Computation Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'parameter_space_multi_curve_a{alphabet_size}_c{code_length}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def analyze_computation_tradeoffs(optimal_configs: Dict[int, Dict[str, Any]]) -> None:
    """
    Analyze trade-offs between effective code rate and computation time.
    
    Args:
        optimal_configs: Dictionary mapping alphabet sizes to their optimal configurations
    """
    print("\nAnalyzing computation time trade-offs...")
    
    # Extract data for analysis
    alphabet_sizes = []
    max_code_rates = []
    efficient_code_rates = []
    max_computation_times = []
    efficient_computation_times = []
    max_efficiencies = []
    efficient_efficiencies = []
    
    for alphabet_size, configs in optimal_configs.items():
        if configs and 'optimal_config' in configs and 'most_efficient_config' in configs:
            optimal = configs['optimal_config']
            efficient = configs['most_efficient_config']
            
            if optimal and efficient:
                alphabet_sizes.append(alphabet_size)
                max_code_rates.append(optimal['effective_code_rate'])
                efficient_code_rates.append(efficient['effective_code_rate'])
                max_computation_times.append(optimal['total_computation_time'])
                efficient_computation_times.append(efficient['total_computation_time'])
                max_efficiencies.append(optimal['computational_efficiency'])
                efficient_efficiencies.append(efficient['computational_efficiency'])
                
                print(f"Alphabet {alphabet_size}:")
                print(f"  Max Code Rate Config: Rate={optimal['effective_code_rate']:.3f}, Time={optimal['total_computation_time']:.1f}ms, Efficiency={optimal['computational_efficiency']:.2f}")
                print(f"  Max Efficiency Config: Rate={efficient['effective_code_rate']:.3f}, Time={efficient['total_computation_time']:.1f}ms, Efficiency={efficient['computational_efficiency']:.2f}")
                
                # Calculate differences
                code_rate_diff = (optimal['effective_code_rate'] - efficient['effective_code_rate']) / optimal['effective_code_rate'] * 100
                time_diff = (optimal['total_computation_time'] - efficient['total_computation_time']) / optimal['total_computation_time'] * 100
                efficiency_diff = (efficient['computational_efficiency'] - optimal['computational_efficiency']) / optimal['computational_efficiency'] * 100
                
                print(f"  Code Rate Difference: {code_rate_diff:.1f}%")
                print(f"  Time Difference: {time_diff:.1f}%")
                print(f"  Efficiency Gain: {efficiency_diff:.1f}%")
    
    # Create comprehensive trade-off visualization
    if alphabet_sizes:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Computational Efficiency vs Code Rate Trade-off Analysis', fontsize=16)
        
        # 1. Code rates comparison
        x_pos = np.arange(len(alphabet_sizes))
        width = 0.35
        ax1.bar(x_pos - width/2, max_code_rates, width, label='Max Code Rate Config', color='blue', alpha=0.7)
        ax1.bar(x_pos + width/2, efficient_code_rates, width, label='Max Efficiency Config', color='green', alpha=0.7)
        ax1.set_xlabel('Alphabet Size')
        ax1.set_ylabel('Effective Code Rate')
        ax1.set_title('Code Rate Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(alphabet_sizes)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Computation times comparison
        ax2.bar(x_pos - width/2, max_computation_times, width, label='Max Code Rate Config', color='blue', alpha=0.7)
        ax2.bar(x_pos + width/2, efficient_computation_times, width, label='Max Efficiency Config', color='green', alpha=0.7)
        ax2.set_xlabel('Alphabet Size')
        ax2.set_ylabel('Computation Time (ms)')
        ax2.set_title('Computation Time Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(alphabet_sizes)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency comparison
        ax3.bar(x_pos - width/2, max_efficiencies, width, label='Max Code Rate Config', color='blue', alpha=0.7)
        ax3.bar(x_pos + width/2, efficient_efficiencies, width, label='Max Efficiency Config', color='green', alpha=0.7)
        ax3.set_xlabel('Alphabet Size')
        ax3.set_ylabel('Computational Efficiency')
        ax3.set_title('Computational Efficiency Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(alphabet_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency vs Code Rate scatter plot
        ax4.scatter(max_code_rates, max_efficiencies, s=100, alpha=0.7, color='blue', label='Max Code Rate Config')
        ax4.scatter(efficient_code_rates, efficient_efficiencies, s=100, alpha=0.7, color='green', label='Max Efficiency Config')
        
        # Add connecting lines
        for i in range(len(alphabet_sizes)):
            ax4.plot([max_code_rates[i], efficient_code_rates[i]], 
                    [max_efficiencies[i], efficient_efficiencies[i]], 
                    'k--', alpha=0.5)
            ax4.annotate(f'A{alphabet_sizes[i]}', 
                        (efficient_code_rates[i], efficient_efficiencies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Effective Code Rate')
        ax4.set_ylabel('Computational Efficiency')
        ax4.set_title('Efficiency vs Code Rate Trade-off')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'computation_tradeoff_analysis.png'), dpi=300)
        plt.close()


def plot_optimal_comparison_with_timing(optimal_configs: Dict[int, Dict[str, Any]]) -> None:
    """
    Create a comparison plot including timing information.
    
    Args:
        optimal_configs: Dictionary mapping alphabet sizes to their configurations
    """
    alphabet_sizes = []
    effective_rates = []
    computation_times = []
    computational_efficiencies = []
    message_lengths = []
    nsym_values = []
    
    for alphabet_size, configs in optimal_configs.items():
        if configs and configs['optimal_config']:
            config = configs['optimal_config']
            alphabet_sizes.append(alphabet_size)
            effective_rates.append(config['effective_code_rate'])
            computation_times.append(config['total_computation_time'])
            computational_efficiencies.append(config['computational_efficiency'])
            message_lengths.append(config['message_length'])
            nsym_values.append(config['nsym'])
    
    if not alphabet_sizes:
        print("No valid configurations to plot.")
        return
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimal Configuration Analysis with Computational Efficiency', fontsize=16)
    
    # 1. Effective code rate vs alphabet size
    ax1.plot(alphabet_sizes, effective_rates, 'o-', linewidth=2, markersize=10, color='blue')
    ax1.set_xlabel('Alphabet Size')
    ax1.set_ylabel('Effective Code Rate')
    ax1.set_title('Optimal Effective Code Rate by Alphabet Size')
    ax1.grid(True, alpha=0.3)
    
    # 2. Computational efficiency vs alphabet size
    ax2.plot(alphabet_sizes, computational_efficiencies, 's-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Alphabet Size')
    ax2.set_ylabel('Computational Efficiency (Code Rate / ms × 1000)')
    ax2.set_title('Computational Efficiency by Alphabet Size')
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter configuration
    x_pos = np.arange(len(alphabet_sizes))
    width = 0.35
    ax3.bar(x_pos - width/2, message_lengths, width, label='Message Length', alpha=0.7)
    ax3.bar(x_pos + width/2, nsym_values, width, label='ECC Symbols', alpha=0.7)
    ax3.set_xlabel('Alphabet Size')
    ax3.set_ylabel('Number of Symbols')
    ax3.set_title('Optimal Parameter Configuration')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(alphabet_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Computation time vs alphabet size
    ax4.plot(alphabet_sizes, computation_times, '^-', linewidth=2, markersize=10, color='red')
    ax4.set_xlabel('Alphabet Size')
    ax4.set_ylabel('Total Computation Time (ms)')
    ax4.set_title('Computation Time by Alphabet Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_configuration_with_timing.png'), dpi=300)
    plt.close()


def main():
    """Main function demonstrating the system comparison and optimization."""
    start_time = time.time()
    
    print("Identification System Comparison & Optimization")
    print("=" * 50)
    print("Including computational efficiency analysis with multiple nsym curves")
    
    # Setup output directory and visualization style
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    utils_old.setup_visualization_style(OUTPUT_DIR)
    
    # Step 1: Find optimal configurations for different alphabet sizes
    print("\nStep 1: Finding optimal configurations for different alphabet sizes...")
    print("Testing multiple nsym curves for comprehensive analysis...")
    
    optimal_configs = {}
    
    for alphabet_size in ALPHABET_SIZES:
        print(f"\n{'-'*50}")
        print(f"ALPHABET SIZE: {alphabet_size}")
        print(f"{'-'*50}")
        
        optimization_result = explore_parameter_space(
            alphabet_size=alphabet_size,
            code_length=1,
            num_trials=DEFAULT_TRIALS,
            num_nsym_curves=NSYM_CURVES
        )
        
        optimal_configs[alphabet_size] = optimization_result
    
    # Step 2: Analyze computation time trade-offs
    print("\nStep 2: Analyzing computational efficiency trade-offs...")
    analyze_computation_tradeoffs(optimal_configs)
    
    # Step 3: Create summary visualizations
    print("\nStep 3: Creating summary visualizations...")
    plot_optimal_comparison_with_timing(optimal_configs)    
    
    # Print comprehensive summary
    print("\nSummary of Optimal Configurations (Across All nsym Curves):")
    print("=" * 50)
    for alphabet_size in sorted(optimal_configs.keys()):
        configs = optimal_configs[alphabet_size]
        if configs and configs['optimal_config']:
            optimal = configs['optimal_config']
            efficient = configs.get('most_efficient_config')
            
            print(f"\nAlphabet Size {alphabet_size}:")
            print(f"  BEST CODE RATE CONFIGURATION:")
            print(f"    Curve: {optimal['curve_idx'] + 1} (nsym fraction: {optimal['nsym_fraction']:.2f})")
            print(f"    Message Length: {optimal['message_length']}")
            print(f"    ECC Symbols: {optimal['nsym']}")
            print(f"    Reliability: {optimal['reliability']:.4f}")
            print(f"    Effective Code Rate: {optimal['effective_code_rate']:.4f}")
            print(f"    Computation Time: {optimal['total_computation_time']:.2f} ms")
            print(f"    Computational Efficiency: {optimal['computational_efficiency']:.2f}")
            
            if efficient and efficient != optimal:
                print(f"  MOST EFFICIENT CONFIGURATION:")
                print(f"    Curve: {efficient['curve_idx'] + 1} (nsym fraction: {efficient['nsym_fraction']:.2f})")
                print(f"    Message Length: {efficient['message_length']}")
                print(f"    ECC Symbols: {efficient['nsym']}")
                print(f"    Reliability: {efficient['reliability']:.4f}")
                print(f"    Effective Code Rate: {efficient['effective_code_rate']:.4f}")
                print(f"    Computation Time: {efficient['total_computation_time']:.2f} ms")
                print(f"    Computational Efficiency: {efficient['computational_efficiency']:.2f}")
        else:
            print(f"\nAlphabet Size {alphabet_size}: No valid configuration found")
    
    # Report execution time
    elapsed = time.time() - start_time
    print(f"\nAnalysis complete. All visualizations saved to {OUTPUT_DIR}.")
    print(f"Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()