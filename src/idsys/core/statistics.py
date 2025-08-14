"""
Statistical utilities for identification system metrics.

This module provides helper functions and classes for statistical 
calculations used in identification system evaluation.
"""

import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

class StatisticsUtils:
    """Utility class for statistical calculations in metrics evaluation."""
    
    @staticmethod
    def merge_pdfs(pdfs_list: List[Dict[Any, float]]) -> Dict[Any, float]:
        """
        Merge multiple probability density functions into a single combined PDF.
        
        Args:
            pdfs_list: List of probability density functions (as dictionaries)
            
        Returns:
            Dictionary representing the merged PDF
        """
        if not pdfs_list:
            return {}
            
        # Count total occurrences across all PDFs
        combined_counts = defaultdict(float)
        total_weight = 0.0
        
        for pdf in pdfs_list:
            # Assume the total weight of each PDF is 1.0
            weight = 1.0 / len(pdfs_list)
            total_weight += weight
            
            for symbol, probability in pdf.items():
                combined_counts[symbol] += probability * weight
        
        # Normalize to create proper PDF
        if total_weight > 0:
            return {symbol: count/total_weight for symbol, count in combined_counts.items()}
        return {}

    @staticmethod
    def calculate_code_rate(num_tags: int, avg_message_length: float, gf_exp: int) -> tuple[float, float]:
        """
        Calculate effective code rate for identification systems.
        
        Code rate is defined as the ratio of log2(log2(N))/output bits,
        where N is the message length and output bits is the tag size.
        
        Args:
            num_tags: Number of tags in the system
            avg_message_length: Average message length in bits
            gf_exp: Galois field exponent (defines tag size)
            
        Returns:
            Tuple of (bulk_code_rate, subsequent_code_rate)
            - bulk_code_rate: Code rate when all tags are sent together
            - subsequent_code_rate: Code rate when tags are sent sequentially
        """
        symbol_size = float(2**gf_exp)
        avg_num_tags = 1  # Average number of tags for subsequent transmission (geometric series)
        if num_tags > 1:
            for i in range(1, num_tags):
                avg_num_tags = avg_num_tags + 1/(symbol_size**i)

        coderate_single = np.log2(np.log2(avg_message_length)) / symbol_size
        coderate_bulk = coderate_single / num_tags
        coderate_subsequently = coderate_single / avg_num_tags

        return coderate_bulk, coderate_subsequently

    @staticmethod
    def calculate_theoretical_fp_rate(k: int, t: int, gf_exp: int) -> float:
        """
        Calculate theoretical false positive rate for k-identification with t tags.
        
        Args:
            k: Number of validation messages
            t: Number of tags
            gf_exp: Exponent for Galois Field size (2^gf_exp)
            
        Returns:
            Theoretical false positive rate
        """
        p = 1.0 / (2 ** gf_exp)  # Base false positive rate for single tag and message
        # For k-identification with t tags: P_fp = 1 - (1 - p^t)^k
        return 1 - (1 - p**t)**k

    @staticmethod
    def combine_hamming_stats(stats1: Dict[str, float], stats2: Dict[str, float]) -> Dict[str, float]:
        """
        Combine two sets of Hamming distance statistics using Welford's algorithm.
        
        Args:
            stats1: First set of statistics with count, mean, M2, min, max
            stats2: Second set of statistics with count, mean, M2, min, max
            
        Returns:
            Combined statistics dictionary
        """
        # If either set is empty, return the other
        if stats1['count'] == 0:
            return stats2.copy()
        if stats2['count'] == 0:
            return stats1.copy()
            
        n1 = stats1['count']
        n2 = stats2['count']
        n = n1 + n2
        
        # Combine means
        delta = stats2['mean'] - stats1['mean']
        combined_mean = (n1 * stats1['mean'] + n2 * stats2['mean']) / n
        
        # Combine M2 values (for variance)
        combined_M2 = stats1['M2'] + stats2['M2'] + delta**2 * n1 * n2 / n
        
        return {
            'count': n,
            'mean': combined_mean,
            'M2': combined_M2,
            'min': min(stats1['min'], stats2['min']),
            'max': max(stats1['max'], stats2['max'])
        }

    @staticmethod
    def calculate_hamming_std(stats: Dict[str, float]) -> float:
        """
        Calculate standard deviation from Hamming statistics.
        
        Args:
            stats: Statistics dictionary with M2 and count
            
        Returns:
            Standard deviation value
        """
        if stats['count'] > 1:
            return math.sqrt(stats['M2'] / stats['count'])
        return 0.0