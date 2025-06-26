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

def run_analysis():
    """Run the analysis."""
    print("=" * 70)
    print("IDENTIFICATION SYSTEM COLLISION ANALYSIS")
    print("=" * 70)
    
    vec_len = 16
    gf_exp = 8
    target_messages = 10**6
    
    patterns = ["random", "incremental", "repeated_patterns", "sparse", "low_entropy"]
    systems = {
        "raw": create_id_system("NoCode", {"gf_exp": gf_exp}),
        "reed_solomon": create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        "rmid": create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 1}),
        "rmid2": create_id_system("RMID", {"gf_exp": gf_exp, "tag_pos": [2], "rm_order": 2}),
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
                num_messages=target_messages,
                show_progress=False
            )
            pattern_results.append(result)
            
            print(f"{system_name:12}: {result['false_positive_rate']:.6f} fpr, "
                  f"{len(result['tag_pdf'].values()):4d} unique tags with {result['total_messages']:4d} messages and {result['num_unique_messages']} unique messages")
            print(f"message set hamming distance: {result['avg_hamming_distance']:.2f}, collisions hamming distance: {result['collisions_avg_hamming_distance']:.2f}")
        
        all_results[pattern] = pattern_results

if __name__ == "__main__":
    run_analysis()


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