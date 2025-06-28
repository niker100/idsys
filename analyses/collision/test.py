#!/usr/bin/env python3
"""
analysis comparing collision behavior between random and structured messages
"""

import sys
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple

# Add framework path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework import create_id_system, IdMetrics

def run_analysis(patterns):
    """Run the analysis."""
    print("=" * 70)
    print("IDENTIFICATION SYSTEM COLLISION ANALYSIS")
    print("=" * 70)
    
    vec_len = 16
    gf_exp = 8
    target_messages = 10**7
    
    systems = {
        "raw": create_id_system("NoCode", {"gf_exp": gf_exp}),
        "reed_solomon": create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),                
        "rmid": create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 1}),
        "rmid10": create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 4}),
        "sha1": create_id_system("SHA1ID", {"gf_exp": gf_exp})
    }
    
    all_results = {}
    
    for pattern in patterns: 

        print("="*40)
        
        pattern_results = []
        for [system_name, system] in systems.items():
            result = IdMetrics.evaluate_system(
                system=system,
                message_pattern=pattern,
                vec_len=vec_len,
                num_messages=target_messages
            )
            pattern_results.append(result)
            
            print(f"{system_name:12}: {result['false_positive_rate']:.6f} fpr, "
                  f"{len(result['tag_pdf'].values()):4d} unique tags with {result['total_messages']:4d} messages")
            print(f"message set hamming distance: {result['avg_hamming_distance']:.2f}, collisions hamming distance: {result['collisions_avg_hamming_distance']:.2f}")
        
        all_results[pattern] = pattern_results

    return all_results

def plot_pdfs(all_results, patterns, systems):
    """Plot the PDFs with theoretical and empirical false positive rates."""
    outdir = "analyses/collision"
    os.makedirs(outdir, exist_ok=True)
    
    def calculate_kl_divergence_from_uniform(pdf_dict, alphabet_size=256):
        """Calculate KL divergence from PDF to uniform distribution using D = log(X) - H(P)"""
        # Calculate entropy H(P)
        entropy = 0.0
        for prob in pdf_dict.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # KL divergence from uniform: D = log(X) - H(P)
        kl_div = np.log2(alphabet_size) - entropy
        return kl_div
    
    def calculate_theoretical_fpr(pdf_dict):
        """Calculate theoretical false positive rate as sum of squared probabilities."""
        return sum(p**2 for p in pdf_dict.values())
    
    for pattern in patterns:
        # Get the message PDF from the first system (same for all systems for a given pattern)
        msg_pdf = all_results[pattern][0]['message_pdf']
        
        # Create arrays for all 256 symbols (0-255)
        all_symbols = list(range(256))
        msg_probs = [msg_pdf.get(symbol, 0.0) for symbol in all_symbols]

        # Create subplots: one extra for message PDF, then one for each system
        fig, axes = plt.subplots(1, len(systems) + 1, figsize=(4 * (len(systems) + 1), 4))
        
        # First subplot: Message PDF
        axes[0].plot(all_symbols, msg_probs, marker='o', linestyle='-', linewidth=2, 
                    markersize=2, color='gray', label='Message PDF')
        axes[0].fill_between(all_symbols, msg_probs, alpha=0.3, color='gray')
        
        # Calculate and display KL divergence for message PDF
        msg_kl_div = calculate_kl_divergence_from_uniform(msg_pdf)
        msg_theoretical_fpr = calculate_theoretical_fpr(msg_pdf)
        axes[0].set_title(f"Message PDF\nKL div: {msg_kl_div:.3f}\nTheoretical FPR: {msg_theoretical_fpr:.6f}")
        axes[0].set_xlabel("Symbol")
        axes[0].set_ylabel("Probability")
        axes[0].set_xlim(0, 255)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Remaining subplots: Tag PDFs for each system
        for idx, (system_name, result) in enumerate(zip(systems, all_results[pattern])):
            tag_pdf = result['tag_pdf']
            tag_probs = [tag_pdf.get(symbol, 0.0) for symbol in all_symbols]
            ax = axes[idx + 1]  # +1 because first subplot is for message PDF
            
            # Plot tag PDF as a line with markers for better visibility
            ax.plot(all_symbols, tag_probs, marker='o', linestyle='-', linewidth=2, 
                   markersize=2, color='C1', label='Tag PDF')
            ax.fill_between(all_symbols, tag_probs, alpha=0.3, color='C1')
            
            # Calculate and display KL divergence and FPRs for tag PDF
            tag_kl_div = calculate_kl_divergence_from_uniform(tag_pdf)
            theoretical_fpr = calculate_theoretical_fpr(tag_pdf)
            empirical_fpr = result['false_positive_rate']
            
            # Add all metrics to the title
            ax.set_title(f"{system_name} Tags\n" +
                        f"KL div: {tag_kl_div:.3f}\n" +
                        f"Theoretical FPR: {theoretical_fpr:.6f}\n" +
                        f"Empirical FPR: {empirical_fpr:.6f}")
            
            ax.set_xlabel("Symbol")
            ax.set_ylabel("Probability")
            ax.set_xlim(0, 255)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f"PDFs for '{pattern}' pattern", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{outdir}/pdf_{pattern}.png", dpi=200)
        plt.close()

def print_fpr_comparison(all_results, patterns, systems):
    """Print a summary table comparing theoretical vs empirical FPRs."""
    print("\n" + "=" * 70)
    print("FALSE POSITIVE RATE COMPARISON: THEORETICAL vs EMPIRICAL")
    print("=" * 70)
    
    def calculate_theoretical_fpr(pdf_dict):
        return sum(p**2 for p in pdf_dict.values())
    
    for pattern in patterns:
        print(f"\n{pattern.upper()}:")
        print(f"{'System':<15} {'Theoretical FPR':<20} {'Empirical FPR':<20} {'Ratio':<10}")
        print("-" * 65)
        
        pattern_results = all_results[pattern]
        for idx, system_name in enumerate(systems):
            result = pattern_results[idx]
            tag_pdf = result['tag_pdf']
            theoretical_fpr = calculate_theoretical_fpr(tag_pdf)
            empirical_fpr = result['false_positive_rate']
            ratio = empirical_fpr / theoretical_fpr if theoretical_fpr > 0 else float('inf')
            
            print(f"{system_name:<15} {theoretical_fpr:<20.6f} {empirical_fpr:<20.6f} {ratio:<10.2f}")

if __name__ == "__main__":
    
    patterns = ["random", "incremental", "repeated_patterns", "sparse", "low_entropy", "only_two"]
    all_results = run_analysis(patterns)
    # After analysis, plot PDFs
    systems = ["raw", "rs", "rm", "rm10", "sha1"]
    # all_results is created in run_analysis()
    plot_pdfs(all_results, patterns, systems)
    print("PDF plots saved in analyses/collision/")

    print_fpr_comparison(all_results, patterns, systems)


# OUTPUT:
# ======================================================================
# IDENTIFICATION SYSTEM COLLISION ANALYSIS
# ======================================================================

# RANDOM MESSAGES:
# ----------------------------------------
# Generated 10000 messages (10000 unique)
# raw         :  745 false positives, 9255 unique tags
# reed_solomon:  702 false positives, 9298 unique tags
# sha1        :  769 false positives, 9231 unique tags

# INCREMENTAL MESSAGES:
# ----------------------------------------
# Generated 10000 messages (10000 unique)
# raw         : 9999 false positives,    1 unique tags
# reed_solomon:    0 false positives, 10000 unique tags
# sha1        :  779 false positives, 9221 unique tags

# REPEATED_PATTERNS MESSAGES:
# ----------------------------------------
# Generated 113 messages (113 unique)
# raw         :  105 false positives,    8 unique tags
# reed_solomon:    0 false positives,  113 unique tags
# sha1        :    0 false positives,  113 unique tags

# SPARSE MESSAGES:
# ----------------------------------------
# Generated 10000 messages (10000 unique)
# raw         : 8749 false positives, 1251 unique tags
# reed_solomon:  695 false positives, 9305 unique tags
# sha1        :  712 false positives, 9288 unique tags

# LOW_ENTROPY MESSAGES:
# ----------------------------------------
# Generated 10000 messages (10000 unique)
# raw         : 9996 false positives,    4 unique tags
# reed_solomon:  710 false positives, 9290 unique tags
# sha1        :  745 false positives, 9255 unique tags