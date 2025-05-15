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
from framework.utils import compare_systems, explore_parameter_effects, create_parameter_optimization_dashboard

def main():
    """Main function demonstrating the identification framework."""
    print("Identification System Analysis Framework")
    print("-" * 50)
    
    # Create output directory for figures
    os.makedirs("output", exist_ok=True)
    os.chdir("output")
    
    # Generate message sets for testing
    message_length = 8
    num_messages = 1024
    string_messages = generate_string_messages(num_messages, message_length)
    # Create identification systems for testing
    systems = {
        "RS-4/8": create_id_system("paper_tagging", {
            "message_length": message_length,
            "nsym": 4,     # number of ECC symbols
            "code_length": 8  # length of tag sequence to extract
        }),
        "RS-8/8": create_id_system("paper_tagging", {
            "message_length": message_length,
            "nsym": 8,     # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
        "RS-16/8": create_id_system("paper_tagging", {
            "message_length": message_length,
            "nsym": 16,      # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
        "RS-32/8": create_id_system("paper_tagging", {
            "message_length": message_length,
            "nsym": 32,     # number of ECC symbols
            "code_length": 8  # length of tag sequence
        }),
    }
    
    # # Compare system performance
    # print("\nStep 1: Comparing system configurations...")
    # code_lengths = [i for i in range(2, 64, 1)]
    # compare_systems(systems, string_messages, code_lengths, num_trials=1000)
    
    # # Parameter effect analysis
    # print("\nStep 2: Analyzing parameter effects...")
    
    # # Analyze effect of ECC symbols (nsym)
    # print("Testing ECC symbol count effect...")
    # rs_system = create_id_system("paper_tagging", {"nsize": 64, "nsym": 8, "code_length": 16})
    # nsym_values = [i for i in range(2, 56, 1)]
    # explore_parameter_effects(rs_system, string_messages, "nsym", nsym_values, num_trials=1000)
    
    # # Analyze effect of code length
    # print("Testing code length effect...")
    # rs_system = create_id_system("paper_tagging", {"nsize": 64, "nsym": 8, "code_length": 8})
    # code_lengths = [i for i in range(2, 64, 1)]
    # explore_parameter_effects(rs_system, string_messages, "code_length", code_lengths, num_trials=1000)
    
    # Create comprehensive parameter optimization dashboard
    print("\nStep 3: Building parameter optimization dashboard...")
    create_parameter_optimization_dashboard(systems, string_messages)
    
    print("\nAnalysis complete. All visualizations saved to the output directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")