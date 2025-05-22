#!/usr/bin/env python3
"""
Single-Symbol Tag Analysis: System Performance Study

This script analyzes the performance characteristics of Reed-Solomon based tagging systems
with a focus on single-symbol tags. It examines several key factors:

1. Error Correction Capacity: How the number of ECC symbols (nsym) affects reliability
   and false positive rates in single-symbol tags.

2. Message Properties: The impact of message length and alphabet size on system 
   performance, including an analysis of message entropy.

3. System Capacity: Understanding how the number of messages affects performance
   and the relationship between message set size and error rates.

The analysis provides insights into optimal parameter selection for tagging systems
and demonstrates the trade-offs between different performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import os
from typing import List, Dict, Any, Optional

from framework import (
    create_id_system, 
    IdMetrics, 
    MessageAnalysisMetrics,
    TaggingMetrics,
    utils
)

# Constants for the analysis
DEFAULT_TRIALS = 2000  # Number of Monte Carlo trials for each measurement
OUTPUT_DIR = "output/single_symbol_tag_analysis"

def analyze_nsym_effect(
    nsym_values: List[int], 
    msg_length: int = 32, 
    num_msgs: int = 100, 
    trials: int = DEFAULT_TRIALS
) -> dict:
    """
    Analyze how the number of error correction symbols affects system performance.
    """
    print("\nAnalyzing effect of error correction capacity (nsym)...")
    
    results = {
        'nsym_values': nsym_values,
        'reliabilities': [],
        'fp_rates': [],
        'effective_code_rates': []
    }
    
    messages = utils.generate_test_messages(num_msgs, msg_length, 4)
    
    for nsym in nsym_values:
        print(f"Testing nsym = {nsym}")
        
        # Create system with specified number of ECC symbols
        system = create_id_system("paper_tagging", {
            "message_length": msg_length,
            "nsym": nsym,
            "code_length": 1  # Single-symbol tag
        })
        
        # Measure performance metrics
        reliability = IdMetrics.reliability(system, messages, trials)
        fp_rate = IdMetrics.error_rates(system, messages, trials)["false_positive_rate"]
        efficiency = IdMetrics.efficiency(system)
        
        print(f"Reliability: {reliability:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"Code Rate: {efficiency['effective_code_rate']:.4f}")
        
        results['reliabilities'].append(reliability)
        results['fp_rates'].append(fp_rate)
        results['effective_code_rates'].append(efficiency['effective_code_rate'])
    
    # Use the new dual-scale plotting function
    utils.plot_performance_metrics_dual_scale(
        metric_values={
            'Reliability': results['reliabilities'],
            'False Positive Rate': results['fp_rates'],
            'Code Rate': results['effective_code_rates']
        },
        x_values=nsym_values,
        x_label='Number of ECC Symbols (nsym)',
        title='Effect of Error Correction Capacity on Performance',
        filename=os.path.join(OUTPUT_DIR, 'nsym_effect.png')
    )
    
    return results

def analyze_message_length(
    lengths: List[int], 
    num_msgs: int = 100, 
    nsym: int = 16,
    trials: int = DEFAULT_TRIALS
) -> dict:
    """
    Analyze how message length affects system performance.
    
    Args:
        lengths: List of message lengths to test
        num_msgs: Number of messages to use in each test
        nsym: Number of error correction symbols
        trials: Number of Monte Carlo trials for each measurement
    
    Returns:
        Dictionary containing all measurement results
    """
    print("\nAnalyzing message length effect...")
    
    results = {
        'lengths': lengths,
        'reliabilities': [],
        'fp_rates': [],
        'effective_code_rates': [],
        'entropies': []
    }
    
    for length in lengths:
        print(f"Testing message length: {length}")
        messages = utils.generate_test_messages(num_msgs, length, 4)
        
        # Create system with single-symbol tag
        system = create_id_system("paper_tagging", {
            "message_length": length,
            "nsym": nsym,
            "code_length": 1
        })
        
        # Measure key metrics
        reliability = IdMetrics.reliability(system, messages, trials)
        fp_rate = IdMetrics.error_rates(system, messages, trials)["false_positive_rate"]
        efficiency = IdMetrics.efficiency(system)
        
        # Calculate message entropy
        char_entropy, msg_entropy = MessageAnalysisMetrics.calculate_message_entropy(messages)
        
        print(f"Reliability: {reliability:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"Code Rate: {efficiency['effective_code_rate']:.4f}")
        print(f"Message Entropy: {msg_entropy:.1f} bits")
        
        results['reliabilities'].append(reliability)
        results['fp_rates'].append(fp_rate)
        results['effective_code_rates'].append(efficiency['effective_code_rate'])
        results['entropies'].append(msg_entropy)
    
    # Create visualization
    utils.plot_performance_metrics_dual_scale(
        metric_values={
            'Reliability': results['reliabilities'],
            'False Positive Rate': results['fp_rates'],
            'Code Rate': results['effective_code_rates']
        },
        x_values=lengths,
        x_label='Message Length (bytes)',
        title='Effect of Message Length on Performance',
        filename=os.path.join(OUTPUT_DIR, 'message_length_effect.png')
    )
    
    # Plot entropy relationship
    plt.figure(figsize=(8, 5))
    plt.plot(lengths, results['entropies'], 'o-', color='purple')
    plt.xlabel('Message Length (bytes)')
    plt.ylabel('Message Entropy (bits)')
    plt.title('Message Entropy vs. Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'message_entropy.png'), dpi=300)
    plt.close()
    
    return results

def analyze_alphabet_size(
    sizes: List[int],
    msg_length: int = 64,
    num_msgs: int = 100,
    nsym: int = 16,
    trials: int = DEFAULT_TRIALS
) -> dict:
    """
    Analyze how alphabet size affects system performance.
    
    Args:
        sizes: List of alphabet sizes to test
        msg_length: Length of each message
        num_msgs: Number of messages to use in each test
        nsym: Number of error correction symbols
        trials: Number of Monte Carlo trials for each measurement
    
    Returns:
        Dictionary containing all measurement results
    """
    print("\nAnalyzing alphabet size effect...")
    
    results = {
        'sizes': sizes,
        'reliabilities': [],
        'fp_rates': [],
        'entropies': []
    }
    
    system = create_id_system("paper_tagging", {
        "message_length": msg_length,
        "nsym": nsym,
        "code_length": 1
    })
    
    for size in sizes:
        print(f"Testing alphabet size: {size}")
        messages = utils.generate_test_messages(num_msgs, msg_length, size)
        
        # Calculate entropy to understand information content
        char_entropy, msg_entropy = MessageAnalysisMetrics.calculate_message_entropy(messages)
        
        # Analyze alphabet usage
        usage_stats = MessageAnalysisMetrics.analyze_alphabet_usage(messages)
        
        # Measure performance metrics
        reliability = IdMetrics.reliability(system, messages, trials)
        fp_rate = IdMetrics.error_rates(system, messages, trials)["false_positive_rate"]
        
        print(f"Reliability: {reliability:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"Character Entropy: {char_entropy:.3f} bits")
        print(f"Most common characters: {usage_stats['most_common']}")
        
        results['reliabilities'].append(reliability)
        results['fp_rates'].append(fp_rate)
        results['entropies'].append(char_entropy)
    
    # Create visualization
    utils.plot_performance_metrics(
        metric_values={
            'Reliability': results['reliabilities'],
            'False Positive Rate': results['fp_rates']
        },
        x_values=sizes,
        x_label='Alphabet Size',
        title='Effect of Alphabet Size on Performance',
        filename=os.path.join(OUTPUT_DIR, 'alphabet_effect.png'),
        log_scale=True
    )
    
    # Create entropy vs performance plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['entropies'], results['reliabilities'], 'o-', label='Reliability')
    plt.plot(results['entropies'], results['fp_rates'], 's-', label='False Positive Rate')
    plt.xlabel('Character Entropy (bits)')
    plt.ylabel('Performance Metric')
    plt.title('Effect of Message Entropy on Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'entropy_performance.png'), dpi=300)
    plt.close()
    
    return results

def analyze_num_messages(
    counts: List[int],
    msg_length: int = 16,
    nsym: int = 8,
    trials: int = DEFAULT_TRIALS
) -> dict:
    """
    Analyze how the number of messages affects system performance.
    
    Args:
        counts: List of message counts to test
        msg_length: Length of each message
        nsym: Number of error correction symbols
        trials: Number of Monte Carlo trials for each measurement
    
    Returns:
        Dictionary containing all measurement results
    """
    print("\nAnalyzing number of messages effect...")
    
    results = {
        'counts': counts,
        'reliabilities': [],
        'fp_rates': [],
        'tag_distributions': []
    }
    
    # Create system with single-symbol tag
    system = create_id_system("paper_tagging", {
        "message_length": msg_length,
        "nsym": nsym,
        "code_length": 1
    })
    
    for count in counts:
        print(f"Testing {count} messages...")
        messages = utils.generate_test_messages(count, msg_length, 4)
        
        # Measure performance metrics
        reliability = IdMetrics.reliability(system, messages, trials)
        fp_rate = IdMetrics.error_rates(system, messages, trials)["false_positive_rate"]
        
        # Analyze tag distribution
        tag_stats = TaggingMetrics.analyze_tag_distribution(system, messages)
        
        print(f"Reliability: {reliability:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print(f"Unique tags: {tag_stats['unique_tags']}")
        print(f"Tag entropy: {tag_stats['tag_entropy']:.2f} bits")
        
        results['reliabilities'].append(reliability)
        results['fp_rates'].append(fp_rate)
        results['tag_distributions'].append(tag_stats)
    
    # Create visualization
    utils.plot_performance_metrics(
        metric_values={
            'Reliability': results['reliabilities'],
            'False Positive Rate': results['fp_rates']
        },
        x_values=counts,
        x_label='Number of Messages',
        title='Effect of Number of Messages on Performance',
        filename=os.path.join(OUTPUT_DIR, 'num_messages_effect.png')
    )
    
    return results

def create_summary_visualization(results_dict: Dict[str, dict]):
    """
    Create a comprehensive and visually appealing summary visualization showing 
    key findings from all analyses using dual-axis plots where appropriate.
    
    Args:
        results_dict: Dictionary containing results from all analyses
    """
    print("\nCreating enhanced summary visualization...")
    
    fig = plt.figure(figsize=(16, 14))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    fig.suptitle('Single-Symbol Tag Performance Analysis', fontsize=18, fontweight='bold')
    
    # 1. ECC Symbols effect (top left) - dual axis plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    nsym_results = results_dict['nsym']
    
    # Plot reliability and FP rate on primary axis
    lns1 = ax1.plot(nsym_results['nsym_values'], nsym_results['reliabilities'], 'o-', 
                    color='navy', linewidth=2, label='Reliability')
    lns2 = ax1.plot(nsym_results['nsym_values'], nsym_results['fp_rates'], 's-', 
                    color='crimson', linewidth=2, label='False Positive Rate')
    
    # Plot code rate on secondary axis
    lns3 = ax1_twin.plot(nsym_results['nsym_values'], nsym_results['effective_code_rates'], '^-', 
                         color='forestgreen', linewidth=2, label='Code Rate')
    
    # Configure axes
    ax1.set_xlabel('Number of ECC Symbols (nsym)')
    ax1.set_ylabel('Reliability / FP Rate')
    ax1_twin.set_ylabel('Effective Code Rate')
    ax1.set_ylim(0, 1.05)
    ax1_twin.tick_params(axis='y', labelcolor='forestgreen')
    
    # Combine legends
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    
    ax1.set_title('Effect of Error Correction Capacity')
    ax1.grid(True, alpha=0.3)
    
    # 2. Message Length effect (top right) - dual axis plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    
    length_results = results_dict['message_length']
    
    # Plot reliability and FP rate on primary axis
    lns1 = ax2.plot(length_results['lengths'], length_results['reliabilities'], 'o-', 
                    color='navy', linewidth=2, label='Reliability')
    lns2 = ax2.plot(length_results['lengths'], length_results['fp_rates'], 's-', 
                    color='crimson', linewidth=2, label='False Positive Rate')
    
    # Plot code rate on secondary axis
    lns3 = ax2_twin.plot(length_results['lengths'], length_results['effective_code_rates'], '^-', 
                         color='forestgreen', linewidth=2, label='Code Rate')
    
    # Configure axes
    ax2.set_xlabel('Message Length (bytes)')
    ax2.set_ylabel('Reliability / FP Rate')
    ax2_twin.set_ylabel('Effective Code Rate')
    ax2.set_ylim(0, 1.05)
    ax2_twin.tick_params(axis='y', labelcolor='forestgreen')
    
    # Combine legends
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')
    
    ax2.set_title('Effect of Message Length')
    ax2.grid(True, alpha=0.3)
    
    # 3. Alphabet Size effect (bottom left) - with entropy subplot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    
    alphabet_results = results_dict['alphabet']
    
    # Plot reliability and FP rate
    lns1 = ax3.plot(alphabet_results['sizes'], alphabet_results['reliabilities'], 'o-', 
                    color='navy', linewidth=2, label='Reliability')
    lns2 = ax3.plot(alphabet_results['sizes'], alphabet_results['fp_rates'], 's-', 
                    color='crimson', linewidth=2, label='False Positive Rate')
    
    # Plot entropy as a line on secondary axis if available
    if 'entropies' in alphabet_results:
        lns3 = ax3_twin.plot(alphabet_results['sizes'], alphabet_results['entropies'], '^-', 
                             color='purple', linewidth=2, label='Entropy (bits)')
        ax3_twin.set_ylabel('Character Entropy (bits)')
        ax3_twin.tick_params(axis='y', labelcolor='purple')
    
    # Configure primary axis
    ax3.set_xlabel('Alphabet Size')
    ax3.set_ylabel('Performance Metric')
    ax3.set_xscale('log', base=2)
    ax3.set_ylim(0, 1.05)
    
    # Combine legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    if 'entropies' in alphabet_results:
        lns += lns3
        labs += [lns3[0].get_label()]
    ax3.legend(lns, labs, loc='best')
    
    ax3.set_title('Effect of Alphabet Size')
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of Messages effect (bottom right) - with tag entropy
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    count_results = results_dict['num_messages']
    
    # Plot reliability and FP rate
    lns1 = ax4.plot(count_results['counts'], count_results['reliabilities'], 'o-', 
                    color='navy', linewidth=2, label='Reliability')
    lns2 = ax4.plot(count_results['counts'], count_results['fp_rates'], 's-', 
                    color='crimson', linewidth=2, label='False Positive Rate')
    
    # Extract and plot tag entropy if available
    if count_results['tag_distributions']:
        tag_entropies = [td['tag_entropy'] for td in count_results['tag_distributions']]
        lns3 = ax4_twin.plot(count_results['counts'], tag_entropies, '^-', 
                             color='purple', linewidth=2, label='Tag Entropy')
        ax4_twin.set_ylabel('Tag Entropy (bits)')
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        ax4_twin.set_ylim(0, max(tag_entropies) * 2)
    
    # Configure primary axis
    ax4.set_xlabel('Number of Messages')
    ax4.set_ylabel('Performance Metric')
    ax4.set_ylim(0, 1.05)
    
    # Combine legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    if count_results['tag_distributions']:
        lns += lns3
        labs += [lns3[0].get_label()]
    ax4.legend(lns, labs, loc='best')
    
    ax4.set_title('Effect of Message Count')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the main title
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close('all')

def main():
    """
    Main function to run all analyses and generate comprehensive insights.
    """
    start_time = time.time()
    utils.setup_visualization_style(OUTPUT_DIR)
    
    # Analysis parameters
    nsym_values = [i for i in range(1, 201, 8)]
    message_lengths = [i for i in range(2, 257, 8)]
    alphabet_sizes = [2, 4, 8, 16, 26, 32, 52, 62, 80, 95, 110, 128, 160, 192, 224, 256]
    msg_counts = [i for i in range(100, 10000, 500)]
    
    # Store all results
    results = {}
    
    # Run analyses
    print("Single-Symbol Tag Performance Analysis")
    print("=" * 40)
    
    print("\nStep 1: Analyzing error correction capacity...")
    results['nsym'] = analyze_nsym_effect(nsym_values)
    
    print("\nStep 2: Analyzing message length effect...")
    results['message_length'] = analyze_message_length(message_lengths)
    
    print("\nStep 3: Analyzing alphabet size effect...")
    results['alphabet'] = analyze_alphabet_size(alphabet_sizes)
    
    print("\nStep 4: Analyzing number of messages effect...")
    results['num_messages'] = analyze_num_messages(msg_counts)
    
    print("\nStep 5: Creating summary visualization...")
    create_summary_visualization(results)
    
    # Calculate and display execution time
    elapsed = time.time() - start_time
    print("\nAnalysis complete. All visualizations saved to the output directory.")
    print(f"Total execution time: {elapsed:.2f} seconds")
    
    # Print summary of key findings
    print("\nKey Findings:")
    print("1. Error Correction: Higher nsym improves reliability but reduces code rate")
    print("2. Message Length: Performance stable until implementation boundaries (~256 bytes)")
    print("3. Alphabet Size: Larger alphabets generally improve reliability")
    print("4. System Capacity: You can handle large message sets without reliability loss, as the produced tags only depend on the number of symbols.")

if __name__ == "__main__":
    main()