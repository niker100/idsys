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
from typing import List, Dict, Any
import os

from framework import (
    create_id_system, IdSystem,
    generate_numeric_messages, generate_string_messages,
    IdMetrics, IdVisualizer
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
    # Create figure for comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Identification System Comparison", fontsize=16)
    
    # Colors for different systems
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    # 1. Compare reliability across different code lengths
    for i, (name, system) in enumerate(systems.items()):
        reliabilities = []
        for code_length in code_lengths:
            system.encoder.set_parameters({"code_length": code_length})
            reliability = IdMetrics.reliability(system, message_set, num_trials)
            reliabilities.append(reliability)
        
        color = colors[i % len(colors)]
        axes[0, 0].plot(code_lengths, reliabilities, marker='o', linestyle='-',
                        color=color, linewidth=2, label=name)
    
    axes[0, 0].set_xlabel('Code Length (bits)')
    axes[0, 0].set_ylabel('Reliability')
    axes[0, 0].set_title('Reliability Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Compare false positive rates
    for i, (name, system) in enumerate(systems.items()):
        fp_rates = []
        for code_length in code_lengths:
            system.encoder.set_parameters({"code_length": code_length})
            error_rates = IdMetrics.error_rates(system, message_set, num_trials)
            fp_rates.append(error_rates["false_positive_rate"])
        
        color = colors[i % len(colors)]
        axes[0, 1].plot(code_lengths, fp_rates, marker='o', linestyle='-',
                        color=color, linewidth=2, label=name)
    
    axes[0, 1].set_xlabel('Code Length (bits)')
    axes[0, 1].set_ylabel('False Positive Rate')
    axes[0, 1].set_title('False Positive Rate Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Compare worst-case collision probability
    for i, (name, system) in enumerate(systems.items()):
        collision_probs = []
        for code_length in code_lengths:
            system.encoder.set_parameters({"code_length": code_length})
            collision_prob = IdMetrics.worst_case_collision_probability(
                system, message_set, sample_size=min(10, len(message_set)), num_trials=10
            )
            collision_probs.append(collision_prob)
        
        color = colors[i % len(colors)]
        axes[1, 0].plot(code_lengths, collision_probs, marker='o', linestyle='-',
                       color=color, linewidth=2, label=name)
    
    axes[1, 0].set_xlabel('Code Length (bits)')
    axes[1, 0].set_ylabel('Worst-Case Collision Probability')
    axes[1, 0].set_title('Collision Probability Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Compare efficiency (code rate)
    for i, (name, system) in enumerate(systems.items()):
        code_rates = []
        for code_length in code_lengths:
            system.encoder.set_parameters({"code_length": code_length})
            efficiency = IdMetrics.efficiency(system)
            code_rates.append(efficiency["code_rate"])
        
        color = colors[i % len(colors)]
        axes[1, 1].plot(code_lengths, code_rates, marker='o', linestyle='-',
                       color=color, linewidth=2, label=name)
    
    axes[1, 1].set_xlabel('Code Length (bits)')
    axes[1, 1].set_ylabel('Code Rate')
    axes[1, 1].set_title('Code Rate Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('system_comparison.png')
    plt.show()


def explore_parameter_effects(
    system: IdSystem,
    message_set: List[Any],
    param_name: str,
    param_values: List[Any],
    num_trials: int = 100,
    set_on_decoder: bool = False
) -> None:
    """
    Explore how a parameter affects different metrics of the system.
    
    Args:
        system: The identification system to evaluate
        message_set: Set of messages to use for testing
        param_name: Name of the parameter to vary
        param_values: List of values to test for the parameter
        num_trials: Number of trials for each metric calculation
        set_on_decoder: If True, set the parameter on the decoder; otherwise, on the encoder
    """
    # Create figure for parameter exploration plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Effect of {param_name} on System Performance", fontsize=16)
    
    # Store metrics
    reliabilities = []
    false_positives = []
    false_negatives = []
    collision_probs = []
    
    # Calculate metrics for each parameter value
    for value in param_values:
        if set_on_decoder:
            system.decoder.set_parameters({param_name: value})
        else:
            system.encoder.set_parameters({param_name: value})
        # Reliability
        reliability = IdMetrics.reliability(system, message_set, num_trials)
        reliabilities.append(reliability)
        # Error rates
        error_rates = IdMetrics.error_rates(system, message_set, num_trials)
        false_positives.append(error_rates["false_positive_rate"])
        false_negatives.append(error_rates["false_negative_rate"])
        # Collision probability
        collision_prob = IdMetrics.worst_case_collision_probability(
            system, message_set, sample_size=min(10, len(message_set)), num_trials=10
        )
        collision_probs.append(collision_prob)
    # Plot results
    axes[0, 0].plot(param_values, reliabilities, 'b-o', linewidth=2)
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Reliability')
    axes[0, 0].set_title(f'Reliability vs {param_name}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(param_values, false_positives, 'r-o', linewidth=2, label='False Positives')
    axes[0, 1].plot(param_values, false_negatives, 'g-s', linewidth=2, label='False Negatives')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Error Rate')
    axes[0, 1].set_title(f'Error Rates vs {param_name}')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[1, 0].plot(param_values, collision_probs, 'm-o', linewidth=2)
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Worst-Case Collision Probability')
    axes[1, 0].set_title(f'Collision Probability vs {param_name}')
    axes[1, 0].grid(True, alpha=0.3)
    # Create a custom plot in the fourth panel
    axes[1, 1].plot(reliabilities, false_positives, 'ro', label='False Positives')
    axes[1, 1].plot(reliabilities, false_negatives, 'go', label='False Negatives')
    for i, value in enumerate(param_values):
        axes[1, 1].annotate(
            f"{param_name}={value}",
            (reliabilities[i], false_positives[i]),
            xytext=(5, 5),
            textcoords="offset points"
        )
    axes[1, 1].set_xlabel('Reliability')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_title('Error Rates vs Reliability Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'parameter_effects_{param_name}.png')
    plt.show()


def create_noise_test(
    system: IdSystem,
    message: Any,
    noise_levels: List[float],
    num_trials: int = 100
) -> None:
    """
    Test system robustness to noise in the codeword.
    
    Args:
        system: The identification system to evaluate
        message: Original message to encode
        noise_levels: List of noise levels (probability of bit flip)
        num_trials: Number of trials for each noise level
    """
    # Encode the original message
    original_codeword = system.send(message)
    codeword_length = len(original_codeword)
    
    # Store results
    success_rates = []
    
    # For each noise level
    for noise_level in noise_levels:
        successes = 0
        
        # Run multiple trials
        for _ in range(num_trials):
            # Add noise to the codeword
            noisy_codeword = original_codeword.copy()
            flip_indices = np.random.random(codeword_length) < noise_level
            noisy_codeword[flip_indices] = 1 - noisy_codeword[flip_indices]
            
            # Try to identify the message
            is_identified = system.receive(noisy_codeword, message)
            if is_identified:
                successes += 1
        
        # Calculate success rate
        success_rate = successes / num_trials
        success_rates.append(success_rate)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, success_rates, 'b-o', linewidth=2)
    plt.xlabel('Noise Level (probability of bit flip)')
    plt.ylabel('Identification Success Rate')
    plt.title('System Robustness to Noise')
    plt.grid(True, alpha=0.3)
    
    # Add a line showing expected success rate based on Hamming distance threshold
    if hasattr(system.decoder, 'parameters') and 'threshold' in system.decoder.parameters:
        threshold = system.decoder.parameters['threshold']
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   label=f'Threshold = {threshold}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('noise_robustness.png')
    plt.show()


def main():
    """Main function demonstrating the identification framework."""
    print("Identification System Framework Example\n")
    
    # Create output directory for figures if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.chdir("output")
    
    # Generate message sets for testing
    print("Generating message sets...")
    num_messages = 50
    numeric_messages = generate_numeric_messages(num_messages)
    string_messages = generate_string_messages(num_messages)
    
    # Create different identification systems
    print("Creating identification systems...")
    systems = {
        "Hash Tagging": create_id_system("hash_tagging", {"code_length": 16}),
        "Random Projection": create_id_system("random_projection", {
            "code_length": 16,
            "max_distance": 2,
            "seed": 42
        }),
        "Hash Tagging (90% threshold)": create_id_system("hash_tagging", {
            "code_length": 16,
            "threshold": 0.9
        })
    }
    
    # Compare system reliability, error rates, and efficiency
    print("Comparing system performance across different code lengths...")
    code_lengths = [i for i in range(2, 33, 1)]
    compare_systems(systems, string_messages, code_lengths, num_trials=1000)
    
    # Explore the effect of code length on a single system
    print("\nExploring the effect of code length on system performance...")
    hash_system = create_id_system("hash_tagging", {"code_length": 8})
    explore_parameter_effects(hash_system, string_messages, "code_length", 
                             code_lengths, num_trials=1000, set_on_decoder=False)
    
    # Explore the effect of threshold on a decoder
    print("\nExploring the effect of threshold on system performance...")
    threshold_system = create_id_system("hash_tagging", {"code_length": 16, "threshold": 1.0})
    thresholds = [i / 10 for i in range(1, 11)]
    explore_parameter_effects(threshold_system, string_messages, "threshold", 
                             thresholds, num_trials=1000, set_on_decoder=True)
    
    # Test robustness to noise
    print("\nTesting system robustness to noise...")
    robust_system = create_id_system("hash_tagging", {"code_length": 16, "threshold": 0.8})
    test_message = string_messages[0]
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    create_noise_test(robust_system, test_message, noise_levels, num_trials=1000)
    
    # Create a comprehensive dashboard for a system
    print("\nCreating a comprehensive dashboard for the system...")
    dashboard_system = create_id_system("hash_tagging", {
        "code_length": 16,
        "threshold": 0.8
    })
    
    fig = IdVisualizer.create_dashboard(
        dashboard_system, string_messages, 
        code_lengths=[4, 8, 12, 16, 24, 32],
        num_trials=100
    )
    
    plt.savefig('system_dashboard.png')
    plt.show()
    
    print("\nExample completed. Outputs saved as PNG files.")


if __name__ == "__main__":
    main()