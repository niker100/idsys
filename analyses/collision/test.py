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
import pandas as pd

# Add framework path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework import create_id_system, IdMetrics
from framework.core import generate_structured_messages

def run_analysis(patterns):
    """Run the analysis."""
    print("=" * 70)
    print("IDENTIFICATION SYSTEM COLLISION ANALYSIS")
    print("=" * 70)
    
    vec_len = 16
    gf_exp = 8
    target_messages = 10**7
    
    systems = {
        "RAW": create_id_system("NoCode", {"gf_exp": gf_exp}),
        "RSID": create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        "RS2ID": create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": [2], "tag_pos_in": [2]}),     
        "RMID": create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 1}),
        "SHA1ID": create_id_system("SHA1ID", {"gf_exp": gf_exp}),
        "SHA256ID": create_id_system("SHA256ID", {"gf_exp": gf_exp})
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

def calculate_kl_divergence_from_uniform(pdf_dict, alphabet_size=256):
    entropy = 0.0
    for prob in pdf_dict.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    kl_div = np.log2(alphabet_size) - entropy
    return kl_div

def plot_pdfs(all_results, patterns, systems):
    """Plot the PDFs with empirical false positive rates and example messages."""
    outdir = "analyses/collision"
    os.makedirs(outdir, exist_ok=True)

    
def save_pdfs_and_examples(all_results, patterns, systems, outdir="analyses/collision"):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for pattern in patterns:
        # Get message PDF and examples
        msg_pdf = all_results[pattern][0]['message_pdf']
        all_symbols = list(range(256))
        msg_probs = [msg_pdf.get(symbol, 0.0) for symbol in all_symbols]
        # Calculate KL divergence for message PDF
        msg_kl_div = calculate_kl_divergence_from_uniform(msg_pdf)
        # Calculate collision probability for message PDF
        msg_collision_prob = float(np.sum(np.square(msg_probs)))
        # Save the false positive rate for each system in a dict
        fp_rates = {}
        for idx, system_name in enumerate(systems):
            fp_rates[system_name] = all_results[pattern][idx]['false_positive_rate']
        vec_len = 16
        gf_exp = 8
        example_gen = generate_structured_messages(
            vec_len=vec_len,
            pattern_type=pattern,
            gf_exp=gf_exp,
            target_count=3,
            generate_first=False
        )
        examples = []
        try:
            for _ in range(3):
                examples.append(next(example_gen))
        except StopIteration:
            pass
        # Calculate theoretical collision probability (sum p^2) for message and tags
        # Calculate G_KL (relative KL divergence gain) for each system
        row = {
            "pattern": pattern,
            "msg_pdf": msg_probs,
            "examples": examples,
            "msg_kl_div": msg_kl_div,
            "msg_collision_prob": msg_collision_prob
        }
        # Add fp rates for each system
        for system_name in systems:
            row[f"fp_rate_{system_name}"] = fp_rates[system_name]
        # Save tag PDFs, KL divergence, collision probability, G_KL for each system
        for idx, system_name in enumerate(systems):
            tag_pdf = all_results[pattern][idx]['tag_pdf']
            tag_probs = [tag_pdf.get(symbol, 0.0) for symbol in all_symbols]
            tag_kl_div = calculate_kl_divergence_from_uniform(tag_pdf)
            tag_collision_prob = float(np.sum(np.square(tag_probs)))
            # G_KL: relative KL divergence gain
            if msg_kl_div > 1e-5:
                g_kl = (msg_kl_div - tag_kl_div) / msg_kl_div
            else:
                g_kl = 0.0
            row[f"tag_pdf_{system_name}"] = tag_probs
            row[f"tag_kl_div_{system_name}"] = tag_kl_div
            row[f"tag_collision_prob_{system_name}"] = tag_collision_prob
            row[f"g_kl_{system_name}"] = g_kl
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "pdfs_and_examples.csv"), index=False)
    print(f"Saved CSV to {os.path.join(outdir, 'pdfs_and_examples.csv')}")


    # Publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'lines.linewidth': 1.3,
        'lines.markersize': 4,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

    for pattern in patterns:
        msg_pdf = all_results[pattern][0]['message_pdf']
        all_symbols = list(range(256))
        msg_probs = [msg_pdf.get(symbol, 0.0) for symbol in all_symbols]
        msg_kl_div = calculate_kl_divergence_from_uniform(msg_pdf)
        msg_collision_prob = float(np.sum(np.square(msg_probs)))

        fig = plt.figure(figsize=(30, 10))
        gs = fig.add_gridspec(2, len(systems) + 1, height_ratios=[0.8, 2.2], hspace=0.4, wspace=0.25)

        # Example messages (top row)
        ax_examples = fig.add_subplot(gs[0, :])
        vec_len = 16
        gf_exp = 8
        example_gen = generate_structured_messages(
            vec_len=vec_len,
            pattern_type=pattern,
            gf_exp=gf_exp,
            target_count=3,
            generate_first=False
        )
        examples = []
        try:
            for _ in range(3):
                examples.append(next(example_gen))
        except StopIteration:
            pass
        if examples:
            example_matrix = np.array(examples)
            im = ax_examples.imshow(example_matrix, cmap='viridis', aspect='auto')
            ax_examples.set_title(f"Example Messages for '{pattern}' Pattern", fontweight='bold', pad=10)
            ax_examples.set_yticks(range(len(examples)))
            ax_examples.set_yticklabels([f"Example {i+1}" for i in range(len(examples))])
            ax_examples.set_xticks(range(0, vec_len, 2))
            ax_examples.set_xticklabels([str(x) for x in range(0, vec_len, 2)])
            cbar = fig.colorbar(im, ax=ax_examples, orientation='horizontal', pad=0.18, fraction=0.08, aspect=30)
            cbar.set_label('Byte Value')
        else:
            ax_examples.text(0.5, 0.5, f"No example messages available for '{pattern}'",
                             ha='center', va='center', fontsize=12)
            ax_examples.set_xticks([])
            ax_examples.set_yticks([])
            ax_examples.axis('off')

        # Message PDF (bottom left)
        ax_msg = fig.add_subplot(gs[1, 0])
        nonzero = np.count_nonzero(msg_probs)
        color = '#2C3E50'
        ax_msg.plot(all_symbols, msg_probs, marker='.', linestyle='-', linewidth=1.1, color=color, label='Message PDF')
        ax_msg.fill_between(all_symbols, msg_probs, alpha=0.15, color=color)
        ax_msg.set_title(f"Message PDF\nKL div: {msg_kl_div:.3f}\nCollision Prob: {msg_collision_prob:.2e}", fontweight='bold', pad=8)
        ax_msg.set_xlabel("Symbol Value")
        ax_msg.set_ylabel("Probability")
        ax_msg.set_ylim(0, max(msg_probs) * 1.15 if nonzero > 1 else 1.05)
        ax_msg.grid(True, alpha=0.3)
        ax_msg.legend(loc='upper right', frameon=False)

        # Tag PDFs for each system (bottom row)
        for idx, (system_name, result) in enumerate(zip(systems, all_results[pattern])):
            tag_pdf = result['tag_pdf']
            tag_probs = [tag_pdf.get(symbol, 0.0) for symbol in all_symbols]
            nonzero_tag = np.count_nonzero(tag_probs)
            tag_kl_div = calculate_kl_divergence_from_uniform(tag_pdf)
            tag_collision_prob = float(np.sum(np.square(tag_probs)))
            if msg_kl_div > 1e-5:
                g_kl = (msg_kl_div - tag_kl_div) / msg_kl_div
            else:
                g_kl = 0.0
            empirical_fpr = result['false_positive_rate']
            ax_tag = fig.add_subplot(gs[1, idx + 1])
            ax_tag.plot(all_symbols, tag_probs, marker='.', linestyle='-', linewidth=1.1, color='#E74C3C', label='Tag PDF')
            ax_tag.fill_between(all_symbols, tag_probs, alpha=0.15, color='#E74C3C')
            ax_tag.set_title(f"{system_name}\nKL div: {tag_kl_div:.3f}\nCollision Prob: {tag_collision_prob:.2e}\nG_KL: {g_kl:.2f}\nEmpirical FPR: {empirical_fpr:.6f}", fontweight='bold', pad=8)
            ax_tag.set_xlabel("Symbol Value")
            if idx == 0:
                ax_tag.set_ylabel("Probability")
            else:
                ax_tag.set_ylabel("")
            ax_tag.set_ylim(0, max(tag_probs) * 1.15 if nonzero_tag > 1 else 1.05)
            ax_tag.grid(True, alpha=0.3)
            ax_tag.legend(loc='upper right', frameon=False)

        plt.suptitle(f"Probability Distribution Functions for '{pattern}' Pattern", fontsize=15, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
        plt.savefig(f"{outdir}/pdf_{pattern}.svg", format='svg')
        plt.close()


if __name__ == "__main__":
    
    patterns = ["random", "incremental", "repeated_patterns", "sparse", "low_entropy", "only_two"]
    all_results = run_analysis(patterns)
    # After analysis, plot PDFs
    systems = ["RAW", "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"]
    # all_results is created in run_analysis()
    plot_pdfs(all_results, patterns, systems)
    print("PDF plots saved in analyses/collision/")

    save_pdfs_and_examples(all_results, patterns, systems)