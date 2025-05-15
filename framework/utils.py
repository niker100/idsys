#!/usr/bin/env python3
"""
Utilities for identification systems.

This module provides utility functions for the identification system framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import time
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from .core import IdSystem
from .metrics import IdMetrics


def setup_plot_style(context="paper", font_scale=1.2):
    """
    Set up a consistent plotting style for all visualizations.
    
    Args:
        context: Seaborn context name (paper, notebook, talk, poster)
        font_scale: Font size scaling factor
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context(context, font_scale=font_scale)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })


def create_comparison_figure(height_ratios=[1.2, 1, 0.8]):
    """
    Create a figure layout for system comparison plots.
    
    Args:
        height_ratios: List of height ratios for the rows
        
    Returns:
        tuple: (figure, list of axes, table axis)
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=height_ratios)
    
    ax1 = fig.add_subplot(gs[0, 0:2])  # Reliability (main metric)
    ax2 = fig.add_subplot(gs[0, 2])     # False positive rate
    ax3 = fig.add_subplot(gs[1, 0])     # Collision probability
    ax4 = fig.add_subplot(gs[1, 1])     # Code rate
    ax5 = fig.add_subplot(gs[1, 2])     # Trade-off plot
    ax_table = fig.add_subplot(gs[2, :])  # Summary table
    ax_table.axis('off')
    
    fig.suptitle("Identification System Performance Comparison", fontsize=18, fontweight='bold', y=0.98)
    
    return fig, [ax1, ax2, ax3, ax4, ax5], ax_table


def plot_reliability(ax, code_lengths, reliabilities, label, color, marker):
    """
    Plot reliability data with consistent styling.
    
    Args:
        ax: Matplotlib axis to plot on
        code_lengths: List of code lengths
        reliabilities: List of reliability values
        label: Label for the legend
        color: Color for the plot line
        marker: Marker style
    """
    ax.plot(code_lengths, reliabilities, marker=marker, linestyle='-',
            color=color, linewidth=2.5, label=label, markersize=8)
    ax.set_xlabel('Code Length (bytes)')
    ax.set_ylabel('Reliability')
    ax.set_title('System Reliability vs Code Length', fontweight='bold')
    
    # True logarithmic scale for reliability focusing on the critical region
    ax.set_yscale('log')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, which='both')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.4f}'.format(y)))
    
    # Optional - add 0.95 reliability threshold line
    add_threshold_line(ax, 0.95, 'Reliability = 0.95', code_lengths[0])


def plot_false_positive_rate(ax, code_lengths, fp_rates, label, color, marker, max_fp_rate=None):
    """
    Plot false positive rate data with consistent styling.
    
    Args:
        ax: Matplotlib axis to plot on
        code_lengths: List of code lengths
        fp_rates: List of false positive rate values
        label: Label for the legend
        color: Color for the plot line
        marker: Marker style
        max_fp_rate: Maximum false positive rate across all systems (for scaling)
    """
    ax.plot(code_lengths, fp_rates, marker=marker, linestyle='-',
            color=color, linewidth=2.5, label=label, markersize=8)
    ax.set_xlabel('Code Length (bytes)')
    ax.set_ylabel('False Positive Rate')
    ax.set_title('False Positive Rate', fontweight='bold')
    
    if max_fp_rate is not None:
        ax.set_ylim(0, min(1.05, max_fp_rate * 1.1))
    else:
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Optional - add 0.1 FP rate threshold line
    add_threshold_line(ax, 0.1, 'FP Rate = 0.1', code_lengths[0])


def plot_collision_probability(ax, code_lengths, collision_probs, label, color, marker):
    """
    Plot collision probability data with consistent styling.
    
    Args:
        ax: Matplotlib axis to plot on
        code_lengths: List of code lengths
        collision_probs: List of collision probability values
        label: Label for the legend
        color: Color for the plot line
        marker: Marker style
    """
    ax.semilogy(code_lengths, collision_probs, marker=marker, linestyle='-',
                color=color, linewidth=2.5, label=label, markersize=8)
    ax.set_xlabel('Code Length (bytes)')
    ax.set_ylabel('Collision Probability (log scale)')
    ax.set_title('Collision Probability', fontweight='bold')
    ax.set_ylim(0.001, 1.05)
    ax.grid(True, alpha=0.3, which='both')


def plot_code_rate(ax, code_lengths, effective_rates, label, color, marker):
    """
    Plot code rate data with consistent styling.
    
    Args:
        ax: Matplotlib axis to plot on
        code_lengths: List of code lengths
        effective_rates: List of effective code rate values
        label: Label for the legend
        color: Color for the plot line
        marker: Marker style
    """
    ax.plot(code_lengths, effective_rates, marker=marker, linestyle='-',
            color=color, linewidth=2.5, label=label, markersize=8)
    ax.set_xlabel('Code Length (bytes)')
    ax.set_ylabel('Effective Code Rate')
    ax.set_title('Effective Code Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)


def plot_tradeoff(ax, x_values, y_values, sizes, label, color, marker):
    """
    Plot a tradeoff scatter plot with consistent styling.
    
    Args:
        ax: Matplotlib axis to plot on
        x_values: X-axis values (typically reliability)
        y_values: Y-axis values (typically effective code rate)
        sizes: Marker sizes (typically proportional to code length)
        label: Label for the legend
        color: Color for the plot markers
        marker: Marker style
    """
    ax.scatter(x_values, y_values, s=sizes, c=[color], marker=marker, 
              label=label, alpha=0.7, edgecolor='w', linewidth=0.5)
    ax.set_xlabel('Reliability')
    ax.set_ylabel('Effective Code Rate')
    ax.set_title('Reliability vs Code Rate Trade-off', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)


def add_point_annotations(ax, x_values, y_values, annotations, modulo=3):
    """
    Add annotations to points on a plot.
    
    Args:
        ax: Matplotlib axis to add annotations to
        x_values: X-coordinate values
        y_values: Y-coordinate values
        annotations: List of annotation texts
        modulo: Only annotate every nth point (to reduce clutter)
    """
    for i, (x, y, text) in enumerate(zip(x_values, y_values, annotations)):
        if i % modulo == 0:  # Add labels every n points to avoid clutter
            ax.annotate(
                f"{text}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8
            )


def add_threshold_line(ax, y_value, text, x_position, color='r'):
    """
    Add a threshold line with label to a plot.
    
    Args:
        ax: Matplotlib axis to add the line to
        y_value: Y-coordinate for the horizontal line
        text: Text to display by the line
        x_position: X-coordinate for the text
        color: Line and text color
    """
    ax.axhline(y=y_value, color=color, linestyle='--', alpha=0.7, linewidth=1)
    ax.text(x_position, y_value, text, 
             color=color, va='bottom', ha='left', fontsize=10, alpha=0.7)


def create_summary_table(ax, system_names, data_dict):
    """
    Create a styled summary table for system comparison.
    
    Args:
        ax: Matplotlib axis to place the table on
        system_names: List of system names
        data_dict: Dictionary with metric data for each system
        
    Returns:
        The created table object
    """
    # Generate summary data for enhanced table
    data_rows = []
    metrics = ["Avg. Reliability", "Avg. FP Rate", "Avg. Collision Prob.", "Avg. Code Rate"]
    
    for name in system_names:
        data = data_dict[name]
        avg_rel = np.mean(data["reliabilities"])
        avg_fp = np.mean(data["fp_rates"])
        avg_coll = np.mean(data["collision_probs"])
        avg_rate = np.mean(data["effective_rates"])
        data_rows.append([f"{avg_rel:.4f}", f"{avg_fp:.4f}", f"{avg_coll:.4f}", f"{avg_rate:.4f}"])
    
    # Create enhanced summary table
    table = ax.table(
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
    
    # Enhanced table styling
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        elif key[1] == -1:  # First column (system names)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9E1F2')
        else:
            cell.set_facecolor('#E9EDF4')
            # Highlight good/bad values with subtle color
            if key[1] == 0:  # Reliability (higher is better)
                val = float(cell.get_text().get_text())
                if val > 0.95:
                    cell.set_facecolor('#D5E8D4')  # Light green
            elif key[1] == 1 or key[1] == 2:  # FP Rate and Collision (lower is better)
                val = float(cell.get_text().get_text())
                if val < 0.1:
                    cell.set_facecolor('#D5E8D4')  # Light green
                elif val > 0.3:
                    cell.set_facecolor('#F8CECC')  # Light red
    
    return table


def add_optimal_length_notes(fig, optimal_lengths, code_lengths):
    """
    Add optimal code length notes to the figure.
    
    Args:
        fig: Figure to add notes to
        optimal_lengths: Dictionary mapping system names to optimal code lengths
        code_lengths: List of all code lengths tested
    """
    if optimal_lengths:
        optimal_text = "Optimal code lengths (reliability ≥ 0.95, FP rate < 0.1):\n"
        for name, length in optimal_lengths.items():
            optimal_text += f"• {name}: {length} bytes\n"
        fig.text(0.02, 0.02, optimal_text, fontsize=10, va='bottom', ha='left', 
                 bbox=dict(facecolor='#F5F5F5', edgecolor='#CCCCCC', boxstyle='round,pad=0.5'))


def compare_systems(
    systems: Dict[str, IdSystem],
    message_set: List[Any],
    code_lengths: List[int],
    num_trials: int = 100
) -> None:
    """
    Compare multiple identification systems across different metrics with enhanced visualizations.
    
    Args:
        systems: Dictionary mapping system names to IdSystem instances
        message_set: Set of messages to use for testing
        code_lengths: List of code lengths to test
        num_trials: Number of trials for each metric calculation
    """
    # Setup visualization with modern style
    setup_plot_style()
    
    # Create figure with custom layout
    fig, axes, ax_table = create_comparison_figure()
    ax1, ax2, ax3, ax4, ax5 = axes
    
    # Modern color palette for systems
    colors = sns.color_palette("viridis", n_colors=len(systems))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    
    print("\nComparing identification systems...")
    comparison_data = {}
    
    # Collect data with progress tracking
    for i, (name, system) in enumerate(systems.items()):
        print(f"Testing system: {name}")
        reliabilities = []
        fp_rates = []
        collision_probs = []
        code_rates = []
        effective_rates = []
        
        for j, code_length in enumerate(code_lengths):
            print(f"  Testing code length {code_length} bytes ({j+1}/{len(code_lengths)})", end='\r')
            
            # Set parameters for both encoder and decoder
            system.encoder.set_parameters({"code_length": code_length})
            system.decoder.set_parameters({"code_length": code_length})
            
            # Calculate all metrics with refined methods
            reliability = IdMetrics.reliability(system, message_set, num_trials)
            error_rates = IdMetrics.error_rates(system, message_set, num_trials)
            collision_prob = IdMetrics.worst_case_collision_probability(
                system, message_set, sample_size=min(10, len(message_set)), num_trials=10
            )
            efficiency = IdMetrics.efficiency(system)
            
            # Store results
            reliabilities.append(reliability)
            fp_rates.append(error_rates["false_positive_rate"])
            collision_probs.append(collision_prob)
            code_rates.append(efficiency["code_rate"])
            effective_rates.append(efficiency.get("effective_code_rate", efficiency["code_rate"]))
        
        print(f"  Completed testing system: {name}" + " " * 20)
        
        # Store all data for this system
        comparison_data[name] = {
            "reliabilities": reliabilities,
            "fp_rates": fp_rates,
            "collision_probs": collision_probs,
            "code_rates": code_rates,
            "effective_rates": effective_rates
        }
        
        # Plot data for this system with enhanced styling
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Use our utility functions to create consistent plots
        plot_reliability(ax1, code_lengths, reliabilities, name, color, marker)
        plot_false_positive_rate(ax2, code_lengths, fp_rates, name, color, marker)
        plot_collision_probability(ax3, code_lengths, collision_probs, name, color, marker)
        plot_code_rate(ax4, code_lengths, effective_rates, name, color, marker)
        
        # Trade-off plot with marker sizes indicating code length
        sizes = [20 + cl for cl in code_lengths]
        plot_tradeoff(ax5, reliabilities, effective_rates, sizes, name, color, marker)
    
    # Add legends to multi-system plots
    ax1.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
    
    # Add code length annotations to trade-off points
    for name, data in comparison_data.items():
        add_point_annotations(ax5, data["reliabilities"], data["effective_rates"], code_lengths)
    
    # Create summary table
    create_summary_table(ax_table, list(systems.keys()), comparison_data)
    
    # Add detailed insights based on data
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
    
    # Add optimal length notes
    add_optimal_length_notes(fig, optimal_lengths, code_lengths)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save high-quality output
    plt.savefig('system_comparison.png', dpi=300, bbox_inches='tight')
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
    setup_plot_style("notebook", 1.1)
    
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
    collision_probs = []
    effective_rates = []
    
    # Calculate all metrics for each parameter value
    for value in param_values:
        # Set parameters on both encoder and decoder or just the encoder
        system.encoder.set_parameters({param_name: value})
        if set_on_decoder:
            system.decoder.set_parameters({param_name: value})
        
        # Calculate metrics
        reliability = IdMetrics.reliability(system, message_set, num_trials)
        error_rates = IdMetrics.error_rates(system, message_set, num_trials)
        collision_prob = IdMetrics.worst_case_collision_probability(
            system, message_set, sample_size=min(10, len(message_set)), num_trials=10
        )
        efficiency = IdMetrics.efficiency(system)
        
        # Store results
        reliabilities.append(reliability)
        false_positives.append(error_rates["false_positive_rate"])
        collision_probs.append(collision_prob)
        effective_rates.append(efficiency.get("effective_code_rate", efficiency["code_rate"]))
        
        print(f"Value {value}: Reliability = {reliability:.4f}, FP Rate = {error_rates['false_positive_rate']:.4f}")
    
    # Use unified plotting functions for consistent style
    color = '#4472C4'
    plot_reliability(ax1, param_values, reliabilities, param_name, color, 'o')
    ax1.set_xlabel(param_name)  # Override the default label
    
    plot_false_positive_rate(ax2, param_values, false_positives, param_name, '#ED7D31', 'o')
    ax2.set_xlabel(param_name)  # Override the default label
    
    plot_collision_probability(ax3, param_values, collision_probs, param_name, '#ED7D31', 'o')
    ax3.set_xlabel(param_name)  # Override the default label
    
    # Custom trade-off plot
    scatter = ax4.scatter(reliabilities, effective_rates, c=param_values, 
                         cmap='viridis', s=100, alpha=0.7)
    # Add parameter value labels
    add_point_annotations(ax4, reliabilities, effective_rates, param_values, modulo=1)
    
    ax4.set_xlabel('Reliability')
    ax4.set_ylabel('Effective Code Rate')
    ax4.set_title('Reliability vs Code Rate Trade-off', fontweight='bold')
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
    best_rate = effective_rates[best_idx]
    
    note_text = (f"Best {param_name} value: {best_value}\n"
                f"Reliability: {best_reliability:.4f}\n"
                f"FP Rate: {best_fp:.4f}\n"
                f"Code Rate: {best_rate:.4f}")
    
    fig.text(0.02, 0.02, note_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'parameter_effects_{param_name}.png', dpi=300)
    print(f"Parameter analysis completed and saved as 'parameter_effects_{param_name}.png'")



def create_parameter_optimization_dashboard(systems: Dict[str, IdSystem], 
                                           message_set: List[Any],
                                           reliability_threshold: float = 0.95):
    """
    Create a comprehensive dashboard showing optimal parameter combinations
    that maximize code rate while meeting a minimum reliability requirement.

    Args:
        systems: Dictionary mapping system names to IdSystem instances
        message_set: Set of messages to use for testing
        reliability_threshold: Minimum required reliability (default: 0.95)
    """
    # Setup visualization
    setup_plot_style("notebook", 1.1)

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])  # nsym vs code_length for reliability
    ax2 = fig.add_subplot(gs[0, 1])  # nsym vs code_length for code rate
    ax3 = fig.add_subplot(gs[0, 2])  # Constrained optimal solutions
    ax4 = fig.add_subplot(gs[1, 0])  # nsize vs code_length for reliability (now just for illustration)
    ax5 = fig.add_subplot(gs[1, 1])  # nsize vs code_length for code rate (now just for illustration)
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')  # 3D trade-off visualization

    fig.suptitle(f"Parameter Optimization for Maximum Code Rate with Reliability ≥ {reliability_threshold}", 
                fontsize=16, fontweight='bold')

    # Define parameter ranges
    code_lengths = [i for i in range(1, 17, 1)]
    nsym_values = [i for i in range(0, 17, 1)]

    # Prepare data structures for heatmaps
    rel_data_nsym = np.zeros((len(nsym_values), len(code_lengths)))
    rate_data_nsym = np.zeros((len(nsym_values), len(code_lengths)))
    nsize_debug = np.zeros((len(nsym_values), len(code_lengths)))  # For debug printing

    # Base system for testing
    from framework import create_id_system
    base_system = create_id_system("paper_tagging", {"nsym": 8, "code_length": 8})

    print(f"\nCreating parameter optimization dashboard (reliability threshold: {reliability_threshold})...")
    print("Testing parameter combinations to optimize code rate...")

    # Test nsym vs code_length combinations
    for i, nsym in enumerate(nsym_values):
        for j, code_length in enumerate(code_lengths):
            base_system.encoder.set_parameters({"nsym": nsym, "code_length": code_length, "message_length": len(message_set[0])})
            base_system.decoder.set_parameters({"nsym": nsym, "code_length": code_length, "message_length": len(message_set[0])})
            # Debug print nsize
            nsize = base_system.encoder.parameters.get("nsize", None)
            nsize_debug[i, j] = nsize if nsize is not None else -1

            reliability = IdMetrics.reliability(base_system, message_set, num_trials=1000)
            efficiency = IdMetrics.efficiency(base_system)
            effective_rate = efficiency.get("effective_code_rate", efficiency["code_rate"])

            rel_data_nsym[i, j] = reliability
            rate_data_nsym[i, j] = effective_rate

    # The nsize vs code_length plots are now just for illustration/debug
    rel_data_nsize = nsize_debug.copy()
    rate_data_nsize = nsize_debug.copy()

    # Create reliability heatmap for nsym vs code_length
    sns.heatmap(rel_data_nsym, annot=False, fmt=".2f", cmap="YlGnBu", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax1)
    ax1.set_title("Reliability: nsym vs code_length", fontweight='bold')
    ax1.set_xlabel("Code Length")
    ax1.set_ylabel("ECC Symbols (nsym)")

    # Create code rate heatmap for nsym vs code_length
    sns.heatmap(rate_data_nsym, annot=False, fmt=".2f", cmap="YlOrRd", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax2)
    ax2.set_title("Code Rate: nsym vs code_length", fontweight='bold')
    ax2.set_xlabel("Code Length")
    ax2.set_ylabel("ECC Symbols (nsym)")

    # Create constraint-based visualization (reliability threshold mask)
    masked_rate_data = np.ma.masked_array(
        rate_data_nsym, 
        mask=(rel_data_nsym < reliability_threshold)
    )

    sns.heatmap(masked_rate_data, annot=False, fmt=".2f", cmap="YlOrRd", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax3,
                cbar_kws={'label': 'Code Rate'})
    ax3.set_title(f"Optimal Code Rate (Reliability ≥ {reliability_threshold})", fontweight='bold')
    ax3.set_xlabel("Code Length")
    ax3.set_ylabel("ECC Symbols (nsym)")

    # For illustration: show nsize as a heatmap (not for optimization)
    sns.heatmap(rel_data_nsize, annot=False, fmt=".0f", cmap="Blues", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax4)
    ax4.set_title("nsize (debug): nsym vs code_length", fontweight='bold')
    ax4.set_xlabel("Code Length")
    ax4.set_ylabel("ECC Symbols (nsym)")

    sns.heatmap(rate_data_nsize, annot=False, fmt=".0f", cmap="Blues", 
                xticklabels=code_lengths, yticklabels=nsym_values, ax=ax5)
    ax5.set_title("nsize (debug): nsym vs code_length", fontweight='bold')
    ax5.set_xlabel("Code Length")
    ax5.set_ylabel("ECC Symbols (nsym)")

    # Find optimal configuration (max code rate with reliability >= threshold)
    valid_configs = np.where(rel_data_nsym >= reliability_threshold)
    if len(valid_configs[0]) > 0:
        valid_rates = rate_data_nsym[valid_configs]
        optimal_idx = np.argmax(valid_rates)
        optimal_i, optimal_j = valid_configs[0][optimal_idx], valid_configs[1][optimal_idx]

        optimal_nsym = nsym_values[optimal_i]
        optimal_code_length = code_lengths[optimal_j]
        optimal_reliability = rel_data_nsym[optimal_i, optimal_j]
        optimal_rate = rate_data_nsym[optimal_i, optimal_j]

        # Mark optimal point on the constrained heatmap
        ax3.add_patch(plt.Rectangle((optimal_j, optimal_i), 1, 1, fill=False, 
                                   edgecolor='green', lw=3))
    else:
        optimal_nsym = None
        optimal_code_length = None
        optimal_reliability = None
        optimal_rate = None

    # 3D visualization
    X, Y = np.meshgrid(code_lengths, nsym_values)
    surf1 = ax6.plot_surface(X, Y, rel_data_nsym, cmap='Blues', alpha=0.7, label='Reliability')
    threshold_plane = np.ones_like(X) * reliability_threshold
    surf2 = ax6.plot_surface(X, Y, threshold_plane, color='r', alpha=0.3)

    if optimal_nsym is not None:
        ax6.scatter([optimal_code_length], [optimal_nsym], [optimal_reliability], 
                  color='green', s=100, marker='*')
        ax6.text(optimal_code_length, optimal_nsym, optimal_reliability + 0.05, 
               f"Optimal: ({optimal_code_length}, {optimal_nsym})", 
               color='green', fontweight='bold')

    ax6.set_xlabel('Code Length')
    ax6.set_ylabel('ECC Symbols (nsym)')
    ax6.set_zlabel('Reliability')
    ax6.set_title('Reliability Surface with Threshold', fontweight='bold')
    ax6.view_init(30, 45)

    # Add a legend placeholder
    ax6.plot([0], [0], [0], 'b-', label='Reliability Surface')
    ax6.plot([0], [0], [0], 'r-', label=f'Threshold ({reliability_threshold})')
    if optimal_nsym is not None:
        ax6.plot([0], [0], [0], 'g*', markersize=10, label='Optimal Solution')
    ax6.legend(loc='upper left')

    # Systems comparison (optional)
    results = []
    for name, test_sys in systems.items():
        params = test_sys.encoder.parameters
        reliability = IdMetrics.reliability(test_sys, message_set, num_trials=50)
        efficiency = IdMetrics.efficiency(test_sys)
        effective_rate = efficiency.get("effective_code_rate", efficiency["code_rate"])

        results.append((name, params.get("nsize", None), params["nsym"], params["code_length"], 
                      reliability, effective_rate))

    # Insight text
    if optimal_nsym is not None:
        insight_text = (
            f"Optimal Configuration for Maximum Code Rate\n"
            f"subject to Reliability ≥ {reliability_threshold}:\n\n"
            f"• nsym = {optimal_nsym}\n"
            f"• code_length = {optimal_code_length}\n"
            f"• Achieved Reliability: {optimal_reliability:.4f}\n"
            f"• Achieved Code Rate: {optimal_rate:.4f}"
        )
    else:
        insight_text = (
            f"No configurations found that meet the\n"
            f"minimum reliability threshold of {reliability_threshold}.\n\n"
            f"Consider lowering the threshold or expanding\n"
            f"the parameter search space."
        )

    fig.text(0.02, 0.02, insight_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Add system comparison information if available
    if results:
        system_text = "Tested Systems:\n\n"
        for name, nsize, nsym, code_length, rel, rate in results:
            meets_threshold = "Y" if rel >= reliability_threshold else "N"
            system_text += f"• {name}: nsym={nsym}, len={code_length}, nsize={nsize}, rel={rel:.2f} {meets_threshold}\n"

        fig.text(0.75, 0.02, system_text, fontsize=9,
                bbox=dict(facecolor='#F5F5F5', edgecolor='#CCCCCC', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('parameter_optimization_coderate.png', dpi=300)

    print("Parameter optimization dashboard created and saved as 'parameter_optimization_coderate.png'")

    # Return optimal parameters if found
    if optimal_nsym is not None:
        return {
            "nsym": optimal_nsym,
            "code_length": optimal_code_length,
            "reliability": optimal_reliability,
            "code_rate": optimal_rate
        }
    return None