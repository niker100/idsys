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

from framework import create_id_system, generate_test_messages


def generate_structured_messages(vec_len: int, pattern_type: str, gf_exp: int, target_count: int = 5000) -> List[List[int]]:
    """Generate unique messages with specified structural patterns."""
    unique_messages: Set[Tuple[int, ...]] = set()

    if pattern_type == "random":
        messages = generate_test_messages(vec_len=vec_len, gf_exp=gf_exp, count=target_count * 2)
        for msg in messages:
            unique_messages.add(tuple(msg))
            if len(unique_messages) >= target_count:
                break

    elif pattern_type == "incremental":
        for i in range(target_count):
            msg = tuple([0] * (vec_len - 1) + [i % 2**gf_exp])
            unique_messages.add(msg)

    elif pattern_type == "repeated_patterns":
        patterns = [[0xAA, 0xBB], [0xFF, 0x00], [0x12, 0x34], [0xCA, 0xFE]]
        for i in range(target_count * 10):
            pattern_idx = i % len(patterns)
            pattern = patterns[pattern_idx]
            shift = (i // len(patterns)) % len(pattern)
            rotated = pattern[shift:] + pattern[:shift]
            # Offset start position for more uniqueness
            offset = (i // (len(patterns) * len(pattern))) % vec_len
            msg = tuple((([0] * offset) + rotated * ((vec_len + len(rotated) - 1) // len(rotated)))[:vec_len])
            unique_messages.add(msg)
            if len(unique_messages) >= target_count:
                break

    elif pattern_type == "sparse":
        for i in range(target_count * 2):
            msg_array = [0] * vec_len
            num_nonzero = 1 + (i % 3)
            positions = [(i + j*7) % vec_len for j in range(num_nonzero)]
            for pos in positions:
                msg_array[pos] = 1 + (i + pos) % 2**gf_exp
            unique_messages.add(tuple(msg_array))
            if len(unique_messages) >= target_count:
                break

    elif pattern_type == "low_entropy":
        alphabet = [0, 1, 2, 3]
        for i in range(target_count * 2):
            np.random.seed(i)
            msg = tuple(np.random.choice(alphabet, size=vec_len))
            unique_messages.add(msg)
            if len(unique_messages) >= target_count:
                break

    return [list(msg) for msg in unique_messages]


def count_false_positives(messages: List[List[int]], tags: List) -> int:
    """Count true false positives: different messages producing the same tag."""
    tag_to_messages = defaultdict(list)
    for msg, tag in zip(messages, tags):
        msg_tuple = tuple(msg)
        tag_to_messages[tag].append(msg_tuple)
    
    false_positives = 0
    for tag, msg_list in tag_to_messages.items():
        unique_messages = set(msg_list)
        if len(unique_messages) > 1:
            # If n unique messages map to same tag, there are (n-1) collisions
            false_positives += len(unique_messages) - 1
    
    return false_positives


def analyze_system(system_name: str, messages: List[List[int]], gf_exp: int = 16) -> Dict:
    """Analyze identification system performance."""
    systems = {
        "raw": lambda msg: msg[0],
        "reed_solomon": create_id_system("RSID", {"gf_exp": gf_exp, "tag_pos": [2]}),
        "sha1": create_id_system("SHA1ID", {"gf_exp": gf_exp})
    }
    
    system = systems[system_name]
    
    if system_name == "raw":
        tags = [system(msg) for msg in messages]
    else:
        tags = []
        for msg in messages:
            tag = system.send(msg)
            if isinstance(tag, list):
                tags.append(tuple(tag))
            else:
                tags.append(tag)
    
    false_positives = count_false_positives(messages, tags)
    unique_tags = len(set(tags))
    
    return {
        'system': system_name,
        'false_positives': false_positives,
        'unique_tags': unique_tags,
        'total_messages': len(messages)
    }


def run_analysis():
    """Run the analysis."""
    print("=" * 70)
    print("IDENTIFICATION SYSTEM COLLISION ANALYSIS")
    print("=" * 70)
    
    vec_len = 16
    gf_exp = 16
    target_messages = 10000
    
    patterns = ["random", "incremental", "repeated_patterns", "sparse", "low_entropy"]
    systems = ["raw", "reed_solomon", "sha1"]
    
    all_results = {}
    
    for pattern in patterns:
        print(f"\n{pattern.upper()} MESSAGES:")
        print("-" * 40)
        
        messages = generate_structured_messages(vec_len, pattern, gf_exp, target_messages)
        unique_count = len(set(tuple(msg) for msg in messages))
        print(f"Generated {len(messages)} messages ({unique_count} unique)")
        
        pattern_results = []
        for system_name in systems:
            result = analyze_system(system_name, messages, gf_exp)
            pattern_results.append(result)
            
            print(f"{system_name:12}: {result['false_positives']:4d} false positives, "
                  f"{result['unique_tags']:4d} unique tags")
        
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