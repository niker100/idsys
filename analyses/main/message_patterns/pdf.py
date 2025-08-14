#!/usr/bin/env python3
"""
analysis comparing collision behavior between random and structured messages
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

from idsys import create_id_system, IdMetrics, generate_structured_messages

def run_analysis(patterns):
    """Run the analysis."""
    print("=" * 70)
    print("IDENTIFICATION SYSTEM COLLISION ANALYSIS")
    print("=" * 70)
    
    vec_len = 16
    gf_exp = 8
    target_messages = 10**5
    
    systems = {
        "RAW": create_id_system("NoCode", {"gf_exp": gf_exp}),
        "RSID": create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [4]}),
        "RS2ID": create_id_system("RS2ID", {"gf_exp": gf_exp, "tag_pos": [4], "tag_pos_in": [3]}),     
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
                num_messages=target_messages,
                calculate_pdfs=True,  # Calculate PDFs for this pattern
            )
            pattern_results.append(result)
            
            print(f"{system_name:12}: {result['false_positive_rate']:.6f} fpr, "
                  f"{len(result['tag_pdf'].values()):4d} unique tags with {result['total_messages']:4d} messages")
            print(f"message set hamming distance: {result['avg_hamming_distance']:.2f}, collisions hamming distance: {result['collisions_avg_hamming_distance']:.2f}")
        
        all_results[pattern] = pattern_results

    return all_results

def calculate_renyi2_entropy(probs: List[float]) -> float:
    """Calculates the Renyi-2 entropy H_2 = -log2(sum(p_i^2))."""
    collision_prob = np.sum(np.square(probs))
    if collision_prob <= 0:
        return 0.0
    return -np.log2(collision_prob)

def process_and_save_results(all_results, patterns, systems, outdir="/output"):
    """
    Processes raw analysis results to calculate metrics (H_2, G_2),
    saves them to a CSV, and returns a processed DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []
    alphabet_size = 256
    log2_q = np.log2(alphabet_size)

    for pattern in patterns:
        # Get message PDF and examples from the first system's result (they are the same for all)
        base_result = all_results[pattern][0]
        msg_pdf_dict = base_result['message_pdf']
        all_symbols = list(range(alphabet_size))
        msg_probs = [msg_pdf_dict.get(symbol, 0.0) for symbol in all_symbols]
        
        # Calculate metrics for the message distribution
        msg_h2 = calculate_renyi2_entropy(msg_probs)
        
        # Generate example messages
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

        # Create the base row for the CSV
        row = {
            "pattern": pattern,
            "msg_pdf": msg_probs,
            "examples": examples,
            "msg_h2": msg_h2,
        }

        # Process each system's results
        for idx, system_name in enumerate(systems):
            result = all_results[pattern][idx]
            tag_pdf_dict = result['tag_pdf']
            tag_probs = [tag_pdf_dict.get(symbol, 0.0) for symbol in all_symbols]
            
            # Calculate metrics for the tag distribution
            tag_h2 = calculate_renyi2_entropy(tag_probs)
            
            # Calculate normalized entropy gain G_2
            if tag_h2 >= msg_h2: # Gain in entropy
                g2 = (tag_h2 - msg_h2) / (log2_q - msg_h2) if log2_q > 0 else 0.0
            else:  # Loss in entropy
                g2 = (tag_h2 - msg_h2) / msg_h2 if msg_h2 > 0 else 0.0

            # Add system-specific data to the row
            row[f"fp_rate_{system_name}"] = result['false_positive_rate']
            row[f"tag_pdf_{system_name}"] = tag_probs
            row[f"tag_h2_{system_name}"] = tag_h2
            row[f"g2_{system_name}"] = g2
            
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "pdfs_and_examples.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    return df


def plot_results(processed_data, systems, outdir="/output"):
    """Plot the PDFs with empirical false positive rates and example messages from processed data."""
    os.makedirs(outdir, exist_ok=True)
    
    # Publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'figure.titlesize': 15, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
        'lines.linewidth': 1.3, 'lines.markersize': 4, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight'
    })

    for _, row in processed_data.iterrows():
        pattern = row['pattern']
        msg_probs = row['msg_pdf']
        msg_h2 = row['msg_h2']
        examples = row['examples']
        
        fig = plt.figure(figsize=(30, 10))
        gs = fig.add_gridspec(2, len(systems) + 1, height_ratios=[0.8, 2.2], hspace=0.4, wspace=0.25)

        # Example messages (top row)
        ax_examples = fig.add_subplot(gs[0, :])
        if examples:
            example_matrix = np.array(examples)
            im = ax_examples.imshow(example_matrix, cmap='viridis', aspect='auto')
            ax_examples.set_title(f"Example Messages for '{pattern}' Pattern", fontweight='bold', pad=10)
            ax_examples.set_yticks(range(len(examples)))
            ax_examples.set_yticklabels([f"Example {i+1}" for i in range(len(examples))])
            ax_examples.set_xticks(range(0, 16, 2))
            ax_examples.set_xticklabels([str(x) for x in range(0, 16, 2)])
            cbar = fig.colorbar(im, ax=ax_examples, orientation='horizontal', pad=0.18, fraction=0.08, aspect=30)
            cbar.set_label('Byte Value')
        else:
            ax_examples.text(0.5, 0.5, f"No example messages available for '{pattern}'",
                             ha='center', va='center', fontsize=12)
            ax_examples.set_xticks([]); ax_examples.set_yticks([]); ax_examples.axis('off')

        # Message PDF (bottom left)
        ax_msg = fig.add_subplot(gs[1, 0])
        all_symbols = list(range(len(msg_probs)))
        color = '#2C3E50'
        ax_msg.plot(all_symbols, msg_probs, marker='.', linestyle='-', linewidth=1.1, color=color, label='Message PDF')
        ax_msg.fill_between(all_symbols, msg_probs, alpha=0.15, color=color)
        ax_msg.set_title(f"Message PDF\nH₂: {msg_h2:.3f}", fontweight='bold', pad=8)
        ax_msg.set_xlabel("Symbol Value"); ax_msg.set_ylabel("Probability")
        ax_msg.set_ylim(0, max(msg_probs) * 1.15 if np.count_nonzero(msg_probs) > 1 else 1.05)
        ax_msg.grid(True, alpha=0.3); ax_msg.legend(loc='upper right', frameon=False)

        # Tag PDFs for each system (bottom row)
        for idx, system_name in enumerate(systems):
            tag_probs = row[f'tag_pdf_{system_name}']
            tag_h2 = row[f'tag_h2_{system_name}']
            g2 = row[f'g2_{system_name}']
            empirical_fpr = row[f'fp_rate_{system_name}']
            
            ax_tag = fig.add_subplot(gs[1, idx + 1])
            ax_tag.plot(all_symbols, tag_probs, marker='.', linestyle='-', linewidth=1.1, color='#E74C3C', label='Tag PDF')
            ax_tag.fill_between(all_symbols, tag_probs, alpha=0.15, color='#E74C3C')
            ax_tag.set_title(f"{system_name}\n G₂: {g2:.2f}\n H₂: {tag_h2:.3f}\nEmpirical FPR: {empirical_fpr:.6f}", fontweight='bold', pad=8)
            ax_tag.set_xlabel("Symbol Value")
            if idx == 0: ax_tag.set_ylabel("Probability")
            else: ax_tag.set_ylabel("")
            ax_tag.set_ylim(0, max(tag_probs) * 1.15 if np.count_nonzero(tag_probs) > 1 else 1.05)
            ax_tag.grid(True, alpha=0.3); ax_tag.legend(loc='upper right', frameon=False)

        plt.suptitle(f"Probability Distribution Functions for '{pattern}' Pattern", fontsize=15, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)
        plt.savefig(f"{outdir}/pdf_{pattern}.svg", format='svg')
        plt.close()


if __name__ == "__main__":
    
    patterns = ["random", "incremental", "repeated_patterns", "sparse", "low_entropy", "only_two"]
    systems = ["RAW", "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"]
    outdir = os.path.join(os.path.dirname(__file__), "output")

    # 1. Run the core analysis
    all_results = run_analysis(patterns)
    
    # 2. Process results, calculate metrics, and save to CSV
    processed_data = process_and_save_results(all_results, patterns, systems, outdir)

    # 3. Generate plots from the processed data
    plot_results(processed_data, systems, outdir)
    print("PDF plots saved")