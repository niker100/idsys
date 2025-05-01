"""
Visualization module for identification system framework.
Provides graphical representations of key metrics and performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os

# Set up the style for plots
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)


class Visualizer:
    """Class for visualizing key metrics and performance of identification systems."""
    
    def __init__(self, output_dir='plots'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_or_show(self, save_path: Optional[str] = None):
        """
        Save or show the current plot.
        
        Args:
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_rates(self, tag_lengths: List[int], 
                         mean_error_rates: List[float],
                         max_error_rates: List[float],
                         save_path: Optional[str] = None):
        """
        Plot mean and maximum error rates across different tag lengths.
        
        Args:
            tag_lengths: List of tag lengths in bits
            mean_error_rates: List of mean error rates
            max_error_rates: List of maximum error rates
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        # Check if we have any non-zero error rates
        has_nonzero_errors = any(rate > 0 for rate in mean_error_rates + max_error_rates)
        
        plt.plot(tag_lengths, mean_error_rates, 'o-', linewidth=2, 
                label='Mean Error Rate', color='#1f77b4')
        plt.plot(tag_lengths, max_error_rates, 's--', linewidth=2, 
                label='Maximum Error Rate', color='#ff7f0e')
        
        plt.xlabel('Tag Length (bits)')
        plt.ylabel('Error Rate')
        plt.title('Error Rates vs Tag Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Only use log scale if we have non-zero error rates
        if has_nonzero_errors:
            plt.yscale('log')
        else:
            plt.ylim(0, 0.1)  # Set a reasonable range for zero errors
            # Add annotation explaining zero error rates
            plt.text(np.mean(tag_lengths), 0.05, 
                    "All error rates are zero for the tested sample", 
                    ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self.save_or_show(save_path)
        
    def plot_reliability(self, tag_lengths: List[int], reliability_data: Dict[str, List[float]],
                        save_path: Optional[str] = None):
        """
        Plot reliability vs tag length for different encoding schemes.
        
        Args:
            tag_lengths: List of tag lengths in bits
            reliability_data: Dictionary mapping encoding schemes to reliability lists
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        colors = sns.color_palette("Set1", len(reliability_data))
        markers = ['o', 's', '^', 'D', '*']
        
        for i, (scheme, reliability_values) in enumerate(reliability_data.items()):
            plt.plot(tag_lengths, reliability_values, 
                    marker=markers[i % len(markers)], 
                    color=colors[i],
                    label=scheme.capitalize(), 
                    linewidth=2)
        
        plt.xlabel('Tag Length (bits)')
        plt.ylabel('Reliability')
        plt.title('Identification System Reliability vs Tag Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.05)  # Reasonable range for reliability
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_metrics_comparison(self, metrics_data: List[Dict[str, Any]], 
                               metrics: List[str] = ['reliability', 'mean_error_rate', 'max_error_rate'],
                               labels: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """
        Plot a comparison of multiple metrics for different configurations.
        
        Args:
            metrics_data: List of dictionaries containing metrics results
            metrics: List of metric names to compare
            labels: Labels for the configurations
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if not labels:
            labels = [f"Config {i+1}" for i in range(len(metrics_data))]
        
        # Extract the data for plotting
        data_for_plot = []
        for i, metrics_dict in enumerate(metrics_data):
            for metric in metrics:
                if metric in metrics_dict:
                    data_for_plot.append({
                        'Configuration': labels[i],
                        'Metric': metric,
                        'Value': metrics_dict[metric]
                    })
        
        # Convert to DataFrame for seaborn plotting
        df = pd.DataFrame(data_for_plot)
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Metric', y='Value', hue='Configuration', data=df)
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Comparison of Metrics Across Configurations')
        plt.legend(title='Configuration')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_confusion_matrix(self, confusion_matrix: Dict[str, int], title: str,
                             save_path: Optional[str] = None):
        """
        Plot a confusion matrix for identification results.
        
        Args:
            confusion_matrix: Dictionary with keys 'true_positives', 'false_positives',
                              'false_negatives', 'true_negatives'
            title: Title for the plot
            save_path: Path to save the plot (if None, plot is displayed)
        """
        # Extract values
        tp = confusion_matrix.get('true_positives', 0)
        fp = confusion_matrix.get('false_positives', 0)
        fn = confusion_matrix.get('false_negatives', 0)
        tn = confusion_matrix.get('true_negatives', 0)
        
        # Create confusion matrix array
        cm = np.array([
            [tp, fn],
            [fp, tn]
        ])
        
        # Calculate total for percentages
        total = tp + fp + fn + tn
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Custom colormap from green to white to red
        cmap = LinearSegmentedColormap.from_list('rg', ["#FF8080", "white", "#8EFF8E"], N=100)
        
        # Plot heatmap
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                        xticklabels=['Match', 'No Match'],
                        yticklabels=['Match', 'No Match'])
        
        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                text = ax.texts[i * 2 + j]
                value = cm[i, j]
                percentage = 100 * value / total if total > 0 else 0
                ax.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                        ha='center', va='center', color='black', fontsize=9)
        
        plt.title(title)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_efficiency_vs_tag_length(self, tag_lengths: List[int], 
                                     message_sizes: List[int],
                                     efficiency_data: Dict[int, Dict[int, float]],
                                     save_path: Optional[str] = None):
        """
        Plot efficiency vs tag length for different message sizes.
        
        Args:
            tag_lengths: List of tag lengths in bits
            message_sizes: List of message sizes in bytes
            efficiency_data: Nested dictionary mapping tag_length -> message_size -> efficiency
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        colors = sns.color_palette("viridis", len(message_sizes))
        markers = ['o', 's', '^', 'D', '*']
        
        for i, msg_size in enumerate(message_sizes):
            efficiencies = [efficiency_data[t][msg_size] for t in tag_lengths]
            plt.plot(tag_lengths, efficiencies, 
                    marker=markers[i % len(markers)], 
                    color=colors[i],
                    label=f'Message size: {msg_size} bytes', 
                    linewidth=2)
        
        plt.xlabel('Tag Length (bits)')
        plt.ylabel('Efficiency (compression factor)')
        plt.title('Identification Efficiency vs Tag Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_security_vs_tag_length(self, tag_lengths: List[int], 
                                  security_data: Dict[int, Dict[str, Any]],
                                  save_path: Optional[str] = None):
        """
        Plot security metrics vs tag length.
        
        Args:
            tag_lengths: List of tag lengths in bits
            security_data: Dictionary mapping tag lengths to security metrics
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        collision_resistances = [security_data[t]['collision_resistance'] for t in tag_lengths]
        
        plt.plot(tag_lengths, collision_resistances, 'o-', linewidth=2, 
                label='Collision Resistance (bits)', color='#1f77b4')
        
        # Add security level annotations
        for i, tag_length in enumerate(tag_lengths):
            if i % 2 == 0:  # Only show annotations for every other point to avoid clutter
                security_level = security_data[tag_length]['comparable_to'].split('-')[0].strip()
                plt.annotate(security_level, 
                            xy=(tag_length, collision_resistances[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)
        
        plt.xlabel('Tag Length (bits)')
        plt.ylabel('Security Level (bits)')
        plt.title('Security Level vs Tag Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_parameter_impact_heatmap(self, tag_lengths: List[int], 
                                     message_sizes: List[int],
                                     data: Dict[str, Dict[int, Dict[int, float]]],
                                     metric: str,
                                     save_path: Optional[str] = None):
        """
        Create a heatmap showing the impact of parameters on a specific metric.
        
        Args:
            tag_lengths: List of tag lengths in bits
            message_sizes: List of message sizes in bytes
            data: Nested dictionary mapping metric -> tag_length -> message_size -> value
            metric: The metric to visualize ('efficiency' or 'reliability_estimate')
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for heatmap
        heatmap_data = []
        for tag_length in tag_lengths:
            for msg_size in message_sizes:
                heatmap_data.append({
                    'Tag Length (bits)': tag_length,
                    'Message Size (bytes)': msg_size,
                    metric.capitalize(): data[metric][tag_length][msg_size]
                })
        
        df = pd.DataFrame(heatmap_data)
        # Fix pivot syntax to use proper pandas method
        df_pivot = df.pivot(index='Message Size (bytes)', 
                           columns='Tag Length (bits)', 
                           values=metric.capitalize())
        
        # Choose appropriate colormap based on metric
        if metric == 'efficiency':
            cmap = 'viridis'  # Higher is better
            fmt = '.2f'
        else:  # reliability
            cmap = 'RdYlGn'  # Red to green, higher is better
            fmt = '.4f'
        
        # Create heatmap
        ax = sns.heatmap(df_pivot, annot=True, fmt=fmt, cmap=cmap, 
                        linewidths=0.5, cbar_kws={'label': metric.capitalize()})
        
        plt.title(f'Impact of Parameters on {metric.capitalize()}')
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_radar_comparison(self, metrics_data: List[Dict[str, Any]], 
                             metrics: List[str] = ['reliability', 'avg_efficiency', 
                                                 'security.collision_resistance', 
                                                 'mean_error_rate', 'avg_processing_time'],
                             labels: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """
        Create a radar chart to compare multiple configurations across key metrics.
        
        Args:
            metrics_data: List of dictionaries containing metrics results
            metrics: List of metric names to compare
            labels: Labels for the configurations
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if not labels:
            labels = [f"Config {i+1}" for i in range(len(metrics_data))]
        
        # Number of metrics
        n = len(metrics)
        
        # Create angle values (in radians)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        metric_labels = []
        normalized_data = []
        
        # Prepare data and labels
        for i, metric in enumerate(metrics):
            metric_name = metric.split('.')[-1].capitalize()
            metric_labels.append(metric_name)
            
            # Extract values for this metric for normalization
            metric_values = []
            for config in metrics_data:
                # Handle nested metrics like security.collision_resistance
                if '.' in metric:
                    parts = metric.split('.')
                    value = config
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = 0
                            break
                else:
                    value = config.get(metric, 0)
                    
                metric_values.append(value)
            
            # Normalization logic based on metric
            if "error" in metric.lower():
                # Lower is better - invert after normalization
                if max(metric_values) > 0:
                    normalized = [1 - (v / max(metric_values)) for v in metric_values]
                else:
                    normalized = [1 for _ in metric_values]
            elif "time" in metric.lower():
                # Lower is better - invert after normalization
                if max(metric_values) > 0:
                    normalized = [1 - (v / max(metric_values)) for v in metric_values]
                else:
                    normalized = [1 for _ in metric_values]
            else:
                # Higher is better
                if max(metric_values) > 0:
                    normalized = [v / max(metric_values) for v in metric_values]
                else:
                    normalized = [0 for _ in metric_values]
            
            for j in range(len(normalized_data), len(normalized)):
                normalized_data.append([])
            
            # Add normalized values to each config's data
            for j, norm_val in enumerate(normalized):
                normalized_data[j].append(norm_val)
        
        # Close the polygon by repeating the first value
        for i in range(len(normalized_data)):
            normalized_data[i] += normalized_data[i][:1]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        
        colors = sns.color_palette("Set1", len(normalized_data))
        
        for i, data in enumerate(normalized_data):
            ax.plot(angles, data, linewidth=2, label=labels[i], color=colors[i])
            ax.fill(angles, data, alpha=0.1, color=colors[i])
        
        # Add metric labels
        plt.xticks(angles[:-1], metric_labels)
        
        # Add radial ticks
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])
        
        plt.title('Comparative Performance Analysis')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        self.save_or_show(save_path)

    def plot_combined_metrics(self, tag_lengths: List[int],
                         mean_error_rates: List[float],
                         reliabilities: List[float],
                         efficiencies: List[float],
                         save_path: Optional[str] = None):
        """
        Create a combined visualization of multiple key metrics.
        
        Args:
            tag_lengths: List of tag lengths in bits
            mean_error_rates: List of mean error rates for each tag length
            reliabilities: List of reliabilities for each tag length
            efficiencies: List of normalized efficiencies for each tag length
            save_path: Path to save the plot (if None, plot is displayed)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot 1: Mean Error Rate
        ax1.plot(tag_lengths, mean_error_rates, 'o-', color='red', linewidth=2)
        ax1.set_ylabel('Mean Error Rate')
        ax1.set_title('Mean Error Rate vs. Tag Length')
        ax1.grid(True, alpha=0.3)
        
        # Check if we have non-zero error rates for log scale
        if any(mean_error_rates):
            ax1.set_yscale('log')
        else:
            ax1.set_ylim(0, 0.1)  # Set a reasonable range for zero errors
            # Add annotation explaining zero error rates
            ax1.text(np.mean(tag_lengths), 0.05, 
                    "All error rates are zero for the tested sample", 
                    ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Reliability
        ax2.plot(tag_lengths, reliabilities, 'o-', color='green', linewidth=2)
        ax2.set_ylabel('Reliability')
        ax2.set_title('Reliability vs. Tag Length')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.9, 1.01)  # Reliability is typically near 1
        
        # Plot 3: Efficiency
        ax3.plot(tag_lengths, efficiencies, 'o-', color='blue', linewidth=2)
        ax3.set_xlabel('Tag Length (bits)')
        ax3.set_ylabel('Normalized Efficiency')
        ax3.set_title('Normalized Efficiency vs. Tag Length')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)  # Normalized between 0 and 1
        
        plt.tight_layout()
        self.save_or_show(save_path)
            
    def plot_security_efficiency_tradeoff(self, tag_lengths: List[int],
                                         security_levels: List[float],
                                         efficiencies: List[float],
                                         labels: Optional[List[str]] = None,
                                         save_path: Optional[str] = None):
        """
        Create a scatter plot showing the tradeoff between security and efficiency.
        
        Args:
            tag_lengths: List of tag lengths in bits
            security_levels: List of security levels (collision resistance)
            efficiencies: List of efficiency values
            labels: Optional list of labels for each point
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 8))
        
        if labels is None:
            labels = [f"{t} bits" for t in tag_lengths]
        
        # Scatter plot with tag length determining size
        sizes = [t/2 for t in tag_lengths]  # Scale down for better visualization
        sc = plt.scatter(security_levels, efficiencies, s=sizes, c=tag_lengths, 
                      alpha=0.6, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Tag Length (bits)')
        
        # Add labels for each point
        for i, label in enumerate(labels):
            plt.annotate(label, (security_levels[i], efficiencies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.xlabel('Security Level (collision resistance bits)')
        plt.ylabel('Efficiency (compression factor)')
        plt.title('Security vs Efficiency Tradeoff')
        plt.grid(True, alpha=0.3)
        
        # Add a reference line showing ideal tradeoff curve
        x = np.array(security_levels)
        if len(x) > 1:
            # Sort for smooth curve
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = np.array(efficiencies)[idx]
            
            plt.plot(x_sorted, y_sorted, 'k--', alpha=0.3, label='Tradeoff curve')
        
        plt.legend()
        plt.tight_layout()
        self.save_or_show(save_path)
    
    def plot_message_scaling(self, message_sizes: List[int],
                           tag_lengths: List[int],
                           metrics: Dict[int, List[float]],
                           metric_name: str,
                           save_path: Optional[str] = None):
        """
        Plot a metric vs message size for different tag lengths.
        
        Args:
            message_sizes: List of message sizes in bytes
            tag_lengths: List of tag lengths to compare
            metrics: Dictionary mapping tag_length -> list of metric values
            metric_name: Name of the metric being plotted
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        colors = sns.color_palette("viridis", len(tag_lengths))
        markers = ['o', 's', '^', 'D', '*']
        
        # Check if we have non-zero values for proper scaling
        has_nonzero_values = False
        for tag_length in tag_lengths:
            if tag_length in metrics:
                if any(v > 0 for v in metrics[tag_length]):
                    has_nonzero_values = True
                    break
        
        # Plot each tag length series
        for i, tag_length in enumerate(tag_lengths):
            if tag_length in metrics:
                plt.plot(message_sizes, metrics[tag_length], 
                        marker=markers[i % len(markers)], 
                        color=colors[i],
                        label=f'Tag length: {tag_length} bits', 
                        linewidth=2)
        
        # Always use log scale for x-axis (message sizes)
        plt.xscale('log')
        
        # Only use log scale for y-axis if appropriate and we have non-zero values
        if ('time' in metric_name.lower() or 'processing' in metric_name.lower()) and has_nonzero_values:
            plt.yscale('log')
        elif not has_nonzero_values:
            # If all values are zero, use linear scale with appropriate range
            if 'error' in metric_name.lower():
                plt.ylim(0, 0.1)
                # Add annotation explaining zero error rates
                plt.text(np.sqrt(min(message_sizes) * max(message_sizes)), 0.05, 
                        f"All {metric_name.lower()} values are zero for the tested sample", 
                        ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
        plt.xlabel('Message Size (bytes)')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs Message Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_or_show(save_path)
        
    def plot_comparative_encoders(self, tag_lengths: List[int],
                                 metric_standard: List[float],
                                 metric_error_detecting: List[float],
                                 metric_name: str,
                                 save_path: Optional[str] = None):
        """
        Compare different encoder types on a specific metric.
        
        Args:
            tag_lengths: List of tag lengths in bits
            metric_standard: List of metric values for standard encoder
            metric_error_detecting: List of metric values for error detecting encoder
            metric_name: Name of the metric being compared
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        width = min(3, tag_lengths[1] - tag_lengths[0]) * 0.35  # Bar width
        
        # Create the grouped bars
        x = np.array(tag_lengths)
        plt.bar(x - width/2, metric_standard, width, label='Standard Encoder', color='#1f77b4')
        plt.bar(x + width/2, metric_error_detecting, width, label='Error-Detecting Encoder', color='#ff7f0e')
        
        plt.xlabel('Tag Length (bits)')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name} by Encoder Type')
        plt.xticks(tag_lengths)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(metric_standard):
            plt.text(tag_lengths[i] - width/2, v + 0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
        for i, v in enumerate(metric_error_detecting):
            plt.text(tag_lengths[i] + width/2, v + 0.02, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        self.save_or_show(save_path)
        
    def plot_metric_tradeoff(self, x_metric: List[float], y_metric: List[float],
                        tag_lengths: List[int], 
                        x_label: str, y_label: str,
                        save_path: Optional[str] = None):
        """
        Plot the tradeoff between two metrics.
        
        Args:
            x_metric: List of values for x-axis metric
            y_metric: List of values for y-axis metric
            tag_lengths: List of tag lengths in bits (for point labels)
            x_label: Label for x-axis 
            y_label: Label for y-axis
            save_path: Path to save the plot (if None, plot is displayed)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create gradient colors based on tag length
        norm = plt.Normalize(min(tag_lengths), max(tag_lengths))
        colors = plt.cm.viridis(norm(tag_lengths))
        
        # Check if values are suitable for log scale and separate zero/non-zero values
        nonzero_indices = []
        zero_indices = []
        
        for i, y in enumerate(y_metric):
            if y > 0:
                nonzero_indices.append(i)
            else:
                zero_indices.append(i)
                
        has_nonzero_y = len(nonzero_indices) > 0
        
        # Create scatter plot with colors mapped to tag lengths
        scatter = ax.scatter(x_metric, y_metric, s=100, c=colors, alpha=0.7)
        
        # Add connecting line
        ax.plot(x_metric, y_metric, 'k--', alpha=0.5)
        
        # Add annotations for each point
        for i, (x, y, tl) in enumerate(zip(x_metric, y_metric, tag_lengths)):
            # Position the label differently for very small values
            if y == 0 or y < 1e-10:
                xytext = (5, 10)  # Position above for zero values
            else:
                xytext = (5, 5)  # Normal position
                
            ax.annotate(f"{tl} bits", (x, y), xytext=xytext, 
                        textcoords="offset points", fontsize=9)
        
        # Handle log scale for error rates
        if y_label.lower().find('error') >= 0:
            if has_nonzero_y:
                # Extract non-zero values
                nonzero_y = [y_metric[i] for i in nonzero_indices]
                min_nonzero = min(nonzero_y)
                
                # Only use log scale if the range is wide enough to benefit from it
                if max(nonzero_y) / min_nonzero > 10:
                    # For log scale, replace zeros with a small value just for display
                    display_y = y_metric.copy()
                    small_value = min_nonzero / 10
                    
                    # Create separate points for zero values at a small visible value
                    if zero_indices:  
                        ax.set_yscale('log')
                        # Plot original non-zero points
                        nonzero_x = [x_metric[i] for i in nonzero_indices]
                        nonzero_y = [y_metric[i] for i in nonzero_indices]
                        nonzero_colors = [colors[i] for i in nonzero_indices]
                        nonzero_tags = [tag_lengths[i] for i in nonzero_indices]
                        
                        # Clear and re-plot only the non-zero points
                        ax.clear()
                        ax.scatter(nonzero_x, nonzero_y, s=100, c=nonzero_colors, alpha=0.7)
                        ax.plot(nonzero_x, nonzero_y, 'k--', alpha=0.5)
                        
                        # Add annotations for non-zero points
                        for i, (x, y, tl) in enumerate(zip(nonzero_x, nonzero_y, nonzero_tags)):
                            ax.annotate(f"{tl} bits", (x, y), xytext=(5, 5),
                                       textcoords="offset points", fontsize=9)
                        
                        # Add zero-value markers at the bottom of the plot with different style
                        zero_x = [x_metric[i] for i in zero_indices]
                        zero_colors = [colors[i] for i in zero_indices]
                        zero_tags = [tag_lengths[i] for i in zero_indices]
                        
                        # Mark zeros at small value with distinct marker
                        ax.scatter(zero_x, [small_value] * len(zero_x), s=100, 
                                  marker='v', c=zero_colors, alpha=0.5,
                                  edgecolors='black')
                        
                        # Add special annotation for zero points
                        for i, (x, tl) in enumerate(zip(zero_x, zero_tags)):
                            ax.annotate(f"{tl} bits (zero)", (x, small_value), 
                                       xytext=(5, 10), textcoords="offset points", 
                                       fontsize=9)
                        
                        # Special annotation explaining zeros
                        ax.text(np.mean(x_metric), small_value * 2, 
                                "▼ Zero values (plotted at visible level)", 
                                ha='center', fontsize=10, 
                                bbox=dict(facecolor='white', alpha=0.8))
                    else:
                        # If no zeros, just use log scale normally
                        ax.set_yscale('log')
                else:
                    # Range not wide enough for log scale benefit, use linear
                    if min(y_metric) == 0:
                        # Set a good range that includes zero
                        ax.set_ylim(0, max(y_metric) * 1.1)
            else:
                # All zeros case - use linear scale with appropriate range
                ax.set_ylim(0, 0.1)
                ax.text(np.mean(x_metric), 0.05, 
                        "All error rates are zero for the tested sample", 
                        ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Finish plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Tradeoff Between {y_label} and {x_label}')
        ax.grid(True, alpha=0.3)
        
        # Add color bar showing tag length
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
        cbar.set_label('Tag Length (bits)')
        
        plt.tight_layout()
        self.save_or_show(save_path)
            
    def plot_parameter_surface(self, tag_lengths: List[int],
                              message_sizes: List[int],
                              data: Dict[str, Dict[int, Dict[int, float]]],
                              metric: str,
                              save_path: Optional[str] = None):
        """
        Create a 3D surface plot showing how parameters affect a metric.
        
        Args:
            tag_lengths: List of tag lengths in bits
            message_sizes: List of message sizes in bytes
            data: Nested dictionary mapping metric -> tag_length -> message_size -> value
            metric: The metric to visualize ('efficiency' or 'reliability_estimate')
            save_path: Path to save the plot (if None, plot is displayed)
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid
        X, Y = np.meshgrid(tag_lengths, message_sizes)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill Z values
        for i, msg_size in enumerate(message_sizes):
            for j, tag_length in enumerate(tag_lengths):
                Z[i, j] = data[metric][tag_length][msg_size]
        
        # Choose colormap based on metric
        cmap = 'viridis' if metric == 'efficiency' else 'RdYlGn'
        
        # Create the surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                            linewidth=0, antialiased=True, alpha=0.8)
        
        # Set the viewing angle for better perspective
        ax.view_init(elev=25, azim=-35)  # Adjust elevation and azimuth for clearer view
        
        # Use log scale for message sizes if spans multiple orders of magnitude
        if max(message_sizes) / min(message_sizes) > 100:
            # Log scale not directly supported in 3D, so adjust tick labels
            log_message_sizes = np.log10(message_sizes)
            tick_positions = np.linspace(min(log_message_sizes), max(log_message_sizes), 5)
            tick_labels = [f"{10**pos:.0f}" for pos in tick_positions]
            ax.set_yticks(10**tick_positions)
            ax.set_yticklabels(tick_labels)
        
        # Improve axis labels with better positioning
        ax.set_xlabel('Tag Length (bits)', labelpad=10)
        ax.set_ylabel('Message Size (bytes)', labelpad=10)
        
        # Format Z-axis label based on metric
        if metric == 'efficiency':
            ax.set_zlabel('Efficiency (compression ratio)', labelpad=10)
        else:
            ax.set_zlabel('Reliability Estimate', labelpad=10)
            
        # Add a more descriptive title
        metric_display = 'Efficiency' if metric == 'efficiency' else 'Reliability'
        ax.set_title(f'Parameter Impact on {metric_display} (3D Surface)', pad=20, fontsize=14)
        
        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        cbar.set_label(metric_display, rotation=90, labelpad=15, fontsize=12)
        
        # Add gridlines for better depth perception
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add a subtle floor projection to help with spatial perception
        ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 0.1 * (np.max(Z) - np.min(Z)),
                   cmap=cmap, alpha=0.3)
                   
        # Ensure proper aspect ratio
        # Calculate ranges for auto-scaling
        x_range = max(tag_lengths) - min(tag_lengths)
        y_range = max(message_sizes) - min(message_sizes)
        z_range = np.max(Z) - np.min(Z)
        
        # Use largest range as reference for scaling
        max_range = max(x_range, y_range, z_range)
        
        # Set axis limits with some padding
        ax.set_xlim(min(tag_lengths) - 0.05 * x_range, max(tag_lengths) + 0.05 * x_range)
        ax.set_ylim(min(message_sizes) - 0.05 * y_range, max(message_sizes) + 0.05 * y_range)
        ax.set_zlim(np.min(Z) - 0.05 * z_range, np.max(Z) + 0.05 * z_range)
        
        plt.tight_layout()
        self.save_or_show(save_path)