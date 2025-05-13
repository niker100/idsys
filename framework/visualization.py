#!/usr/bin/env python3
"""
Visualization module for identification system performance.

This module provides functions and classes for creating
visualizations of identification system performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import itertools

from .core import IdSystem, IdEncoder, IdDecoder
from .metrics import IdMetrics


class IdVisualizer:
    """Class for creating visualizations of identification system performance."""
    
    @staticmethod
    def plot_reliability_vs_code_length(
        system: IdSystem,
        message_set: List[Any],
        code_lengths: List[int],
        num_trials: int = 500,
        ax: Optional[Axes] = None,
        title: str = "Reliability vs. Code Length"
    ) -> Tuple[Figure, Axes]:
        """
        Plot the reliability of the system for different code lengths.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            code_lengths: List of code lengths to test
            num_trials: Number of trials for each code length
            ax: Optional matplotlib Axes to plot on
            title: Plot title
            
        Returns:
            Tuple of (Figure, Axes) with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        reliabilities = []
        
        for code_length in code_lengths:
            # Update code length
            system.encoder.set_parameters({"code_length": code_length})
            
            # Calculate reliability
            reliability = IdMetrics.reliability(system, message_set, num_trials)
            reliabilities.append(reliability)
            
        # Create plot
        ax.plot(code_lengths, reliabilities, marker='o', linestyle='-', linewidth=2)
        ax.set_xlabel('Code Length (bits)')
        ax.set_ylabel('Reliability')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 1.0 (perfect reliability)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Reliability')
        
        # Set y-axis limits
        ax.set_ylim([min(reliabilities) * 0.95, 1.05])
        
        # Add legend
        ax.legend()
        
        return fig, ax
    
    @staticmethod
    def plot_error_rates(
        system: IdSystem,
        message_set: List[Any],
        parameter_values: List[Any],
        parameter_name: str,
        parameter_setter: Optional[callable] = None,
        num_trials: int = 500,
        ax: Optional[Axes] = None,
        title: str = "Error Rates"
    ) -> Tuple[Figure, Axes]:
        """
        Plot false positive and false negative rates for different parameter values.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            parameter_values: List of parameter values to test
            parameter_name: Name of the parameter being varied
            parameter_setter: Function to set the parameter (default: set via encoder.set_parameters)
            num_trials: Number of trials for each parameter value
            ax: Optional matplotlib Axes to plot on
            title: Plot title
            
        Returns:
            Tuple of (Figure, Axes) with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        false_positive_rates = []
        false_negative_rates = []
        
        for value in parameter_values:
            # Update parameter
            if parameter_setter is not None:
                parameter_setter(system, value)
            else:
                system.encoder.set_parameters({parameter_name: value})
            
            # Calculate error rates
            error_rates = IdMetrics.error_rates(system, message_set, num_trials)
            false_positive_rates.append(error_rates["false_positive_rate"])
            false_negative_rates.append(error_rates["false_negative_rate"])
            
        # Create plot
        ax.plot(parameter_values, false_positive_rates, marker='o', linestyle='-', 
                linewidth=2, label='False Positive Rate')
        ax.plot(parameter_values, false_negative_rates, marker='s', linestyle='-', 
                linewidth=2, label='False Negative Rate')
        
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Error Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax
    
    @staticmethod
    def plot_collision_matrix(
        system: IdSystem,
        message_set: List[Any],
        max_messages: int = 20,
        ax: Optional[Axes] = None,
        title: str = "Collision Matrix"
    ) -> Tuple[Figure, Axes]:
        """
        Plot a heatmap showing which messages collide with each other.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            max_messages: Maximum number of messages to include in the matrix
            ax: Optional matplotlib Axes to plot on
            title: Plot title
            
        Returns:
            Tuple of (Figure, Axes) with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
            
        # Get collision matrix
        collision_mat = IdMetrics.collision_matrix(system, message_set, max_messages)
        
        # Create heatmap
        sns.heatmap(collision_mat.astype(int), cmap='YlOrRd', annot=True, fmt="d",
                    cbar_kws={'label': 'Collision (1=Yes, 0=No)'},
                    xticklabels=[str(m) for m in message_set[:max_messages]],
                    yticklabels=[str(m) for m in message_set[:max_messages]],
                    ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Message at Receiver')
        ax.set_ylabel('Message at Sender')
        
        return fig, ax
    
    @staticmethod
    def plot_efficiency_vs_parameter(
        system: IdSystem,
        parameter_values: List[Any],
        parameter_name: str,
        parameter_setter: Optional[callable] = None,
        efficiency_metric: str = "code_rate",
        ax: Optional[Axes] = None,
        title: str = "Efficiency vs. Parameter"
    ) -> Tuple[Figure, Axes]:
        """
        Plot the efficiency metric against a parameter.
        
        Args:
            system: The identification system to evaluate
            parameter_values: List of parameter values to test
            parameter_name: Name of the parameter being varied
            parameter_setter: Function to set the parameter (default: set via encoder.set_parameters)
            efficiency_metric: Which efficiency metric to plot (code_rate or encoding_time_ms)
            ax: Optional matplotlib Axes to plot on
            title: Plot title
            
        Returns:
            Tuple of (Figure, Axes) with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        metric_values = []
        
        for value in parameter_values:
            # Update parameter
            if parameter_setter is not None:
                parameter_setter(system, value)
            else:
                system.encoder.set_parameters({parameter_name: value})
            
            # Calculate efficiency metrics
            metrics = IdMetrics.efficiency(system)
            metric_values.append(metrics[efficiency_metric])
            
        # Create plot
        ax.plot(parameter_values, metric_values, marker='o', linestyle='-', linewidth=2)
        
        xlabel = parameter_name
        ylabel = "Code Rate" if efficiency_metric == "code_rate" else "Encoding Time (ms)"
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_parameter_sweep(
        system: IdSystem,
        message_set: List[Any],
        param1_values: List[Any],
        param1_name: str,
        param2_values: List[Any],
        param2_name: str,
        metric_func: callable,
        metric_name: str,
        num_trials: int = 100,
        cmap: str = 'viridis',
        title: str = "Parameter Sweep"
    ) -> Tuple[Figure, Axes]:
        """
        Create a heatmap showing a metric for different combinations of two parameters.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            param1_values: List of values for first parameter
            param1_name: Name of first parameter
            param2_values: List of values for second parameter
            param2_name: Name of second parameter
            metric_func: Function that computes the metric for a system and message_set
            metric_name: Name of the metric being computed
            num_trials: Number of trials for each parameter combination
            cmap: Colormap to use
            title: Plot title
            
        Returns:
            Tuple of (Figure, Axes) with the plot
        """
        # Create result matrix
        results = np.zeros((len(param1_values), len(param2_values)))
        
        # Calculate metrics for each parameter combination
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                # Update parameters
                system.encoder.set_parameters({
                    param1_name: p1,
                    param2_name: p2
                })
                
                # Calculate metric
                results[i, j] = metric_func(system, message_set, num_trials)
        
        # Create figure and heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(results, cmap=cmap, interpolation='nearest', aspect='auto',
                      extent=[min(param2_values), max(param2_values), 
                             max(param1_values), min(param1_values)])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
        
        # Label axes
        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)
        
        # Add grid
        ax.set_xticks(param2_values)
        ax.set_yticks(param1_values)
        ax.grid(color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def create_dashboard(
        system: IdSystem,
        message_set: List[Any],
        code_lengths: List[int] = None,
        num_trials: int = 100
    ) -> Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            code_lengths: List of code lengths to test (default: [4, 8, 16, 32, 64])
            num_trials: Number of trials for each evaluation
            
        Returns:
            Figure: The dashboard figure
        """
        if code_lengths is None:
            code_lengths = [4, 8, 16, 32, 64]
        
        # Create a 2x2 grid for the dashboard
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("Identification System Performance Dashboard", fontsize=16)
        
        # 1. Reliability vs. Code Length
        IdVisualizer.plot_reliability_vs_code_length(
            system, message_set, code_lengths, num_trials, axes[0, 0],
            title="Reliability vs. Code Length"
        )
        
        # 2. Error Rates vs. Code Length
        IdVisualizer.plot_error_rates(
            system, message_set, code_lengths, "code_length", None, num_trials, axes[0, 1],
            title="Error Rates vs. Code Length"
        )
        
        # 3. Efficiency Metrics
        IdVisualizer.plot_efficiency_vs_parameter(
            system, code_lengths, "code_length", None, "code_rate", axes[1, 0],
            title="Code Rate vs. Code Length"
        )
        
        # 4. Collision Matrix (using a fixed code length)
        system.encoder.set_parameters({"code_length": 8})  # Use a moderate code length
        IdVisualizer.plot_collision_matrix(
            system, message_set, min(10, len(message_set)), axes[1, 1],
            title="Collision Matrix (Code Length = 8)"
        )
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig


class InteractiveIdVisualizer:
    """Class for interactive visualization of identification systems."""
    
    def __init__(self, system: IdSystem, message_generator: callable):
        """
        Initialize the interactive visualizer.
        
        Args:
            system: The identification system to evaluate
            message_generator: Function that generates message sets of a given size
        """
        self.system = system
        self.message_generator = message_generator
        
        # Default parameters
        self.params = {
            "code_length": 8,
            "message_set_size": 100,
            "num_trials": 100
        }
        
    def update_plot(self):
        """Update the visualization with current parameters."""
        # Generate message set
        message_set = self.message_generator(self.params["message_set_size"])
        
        # Update system parameters
        self.system.encoder.set_parameters({"code_length": self.params["code_length"]})
        
        # Create dashboard
        fig = IdVisualizer.create_dashboard(
            self.system, message_set,
            code_lengths=[4, 8, 12, 16, 24, 32],
            num_trials=self.params["num_trials"]
        )
        
        return fig
    
    def create_interactive_dashboard(self):
        """
        Create an interactive dashboard using matplotlib widgets.
        
        Note: This requires implementing matplotlib widgets in a notebook
        or other interactive environment.
        """
        # Note: This is a placeholder for the interactive dashboard.
        # In an actual implementation, you'd use matplotlib widgets,
        # ipywidgets, or a framework like Dash or Streamlit.
        print("Interactive dashboard creation requires a graphical environment.")
        print("Consider implementing this using Jupyter notebooks with ipywidgets,")
        print("or a web framework like Dash or Streamlit.")
        
        # Return a static dashboard as a fallback
        return self.update_plot()