#!/usr/bin/env python3
"""
Utilities for identification systems.

This module provides utility functions for the identification system framework,
including visualization tools and test data generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import string
from typing import List, Dict, Any, Tuple, Optional
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from .core import IdSystem
from .metrics import IdMetrics

DEFAULT_ALPHABETS = {
    2: "01",                                   # Binary
    3: "ABC",                                  # Ternary
    4: "ACTG",                                 # DNA nucleotides
    8: "01234567",                             # Octal
    16: "0123456789ABCDEF",                    # Hexadecimal
    26: string.ascii_lowercase,                # English lowercase
    52: string.ascii_letters,                  # English letters
    62: string.ascii_letters + string.digits,  # Alphanumeric
    95: string.printable.strip()               # Printable ASCII
}

def setup_visualization_style(output_dir: str = "output", context: str = "paper", font_scale: float = 1.2):
    """
    Configure consistent visualization style for all plots.
    
    Args:
        output_dir: Directory to save visualizations
        context: Seaborn context (paper, talk, poster)
        font_scale: Font size scaling factor
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context(context, font_scale=font_scale)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up a color palette for consistent colors across all visualizations
    colors = sns.color_palette("viridis", 4)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

def generate_test_messages(count: int, length: int, alphabet_size: int = 62, seed: Optional[int] = None) -> List[str]:
    """
    Generate test messages with specified characteristics.
    
    Args:
        count: Number of messages to generate
        length: Length of each message
        alphabet_size: Size of alphabet to use (controls entropy)
        seed: Random seed for reproducibility
        
    Returns:
        List of generated messages
    """
    if seed is not None:
        random.seed(seed)
    
    # Select appropriate character set
    if alphabet_size in DEFAULT_ALPHABETS:
        chars = DEFAULT_ALPHABETS[alphabet_size]
    else:
        chars = ''.join(chr(i) for i in range(min(alphabet_size, 256)))
    
    # Generate random messages
    messages = [''.join(random.choice(chars) for _ in range(length))
               for _ in range(count)]
    
    if seed is not None:
        random.seed()
    
    return messages

def plot_performance_metrics(metric_values: Dict[str, List[float]], x_values: List[float], 
                          x_label: str, title: str, filename: str,
                          log_scale: bool = False, threshold_line: Optional[float] = None):
    """
    Create a standardized performance metric plot.
    
    Args:
        metric_values: Dictionary mapping metric names to lists of values
        x_values: X-axis values
        x_label: Label for x-axis
        title: Plot title
        filename: Output filename
        log_scale: Whether to use log scale for x-axis
        threshold_line: Y-value for optional horizontal threshold line
    """
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metric_values.items():
        plt.plot(x_values, values, marker='o', linestyle='-', label=metric_name)
    
    if log_scale:
        plt.xscale('log', base=2)
    
    if threshold_line is not None:
        plt.axhline(y=threshold_line, color='gray', linestyle='--', alpha=0.5,
                   label=f'Threshold ({threshold_line})')
    
    plt.xlabel(x_label)
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_performance_metrics_dual_scale(metric_values, x_values, x_label, title, filename):
    """
    Create visualization with dual y-axes for metrics with different scales.
    
    Args:
        metric_values: Dictionary of metric names and their values
        x_values: X-axis values
        x_label: Label for x-axis
        title: Plot title
        filename: Output file path
    """
    plt.figure(figsize=(10, 6))
    
    # Primary y-axis for metrics in 0-1 range (reliability, FP rate)
    ax1 = plt.gca()
    
    # Secondary y-axis for metrics with larger scale (code rate)
    ax2 = ax1.twinx()
    
    # Plot reliability and FP rate on primary axis
    if 'Reliability' in metric_values:
        ax1.plot(x_values, metric_values['Reliability'], 'o-', 
                color='navy', linewidth=2, label='Reliability')
    
    if 'False Positive Rate' in metric_values:
        ax1.plot(x_values, metric_values['False Positive Rate'], 's-', 
                color='crimson', linewidth=2, label='False Positive Rate')
    
    # Plot code rate on secondary axis
    if 'Code Rate' in metric_values:
        ax2.plot(x_values, metric_values['Code Rate'], '^-', 
                color='forestgreen', linewidth=2, label='Code Rate')
    
    # Configure primary y-axis for reliability metrics (0-1 range)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Reliability / False Positive Rate')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Configure secondary y-axis for code rate
    ax2.set_ylabel('Effective Code Rate')
    ax2.tick_params(axis='y', labelcolor='forestgreen')
    
    # Add grid and finalize plot
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel(x_label)
    plt.title(title)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_analysis_summary(results: Dict[str, Dict[str, List[float]]], 
                          output_dir: str = "output") -> None:
    """
    Create a comprehensive summary visualization of analysis results.
    
    Args:
        results: Dictionary containing results from all analyses
        output_dir: Directory to save the summary visualization
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tagging System Performance Analysis', fontsize=16)
    
    # Plot each analysis in its own subplot
    for (i, j), (name, data) in zip([(0,0), (0,1), (1,0), (1,1)], results.items()):
        ax = axs[i, j]
        x_values = data['x_values']
        
        for metric, values in data['metrics'].items():
            ax.plot(x_values, values, marker='o', label=metric)
        
        ax.set_title(name)
        ax.set_xlabel(data['x_label'])
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if data.get('log_scale', False):
            ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_summary.png'), dpi=300)
    plt.close()

def plot_tradeoff_analysis(x_values: List[float], y_values: List[float],
                          size_values: Optional[List[float]] = None,
                          labels: Optional[List[str]] = None,
                          title: str = "Performance Trade-off Analysis",
                          filename: str = "tradeoff_analysis.png"):
    """
    Create a tradeoff analysis plot, optionally with size-coded points.
    
    Args:
        x_values: Values for x-axis
        y_values: Values for y-axis
        size_values: Optional values for point sizes
        labels: Optional point labels
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 8))
    
    if size_values is not None:
        # Normalize sizes for better visualization
        sizes = [50 + 200 * (s / max(size_values)) for s in size_values]
        sc = plt.scatter(x_values, y_values, s=sizes, alpha=0.6)
    else:
        plt.plot(x_values, y_values, 'o-')
    
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()