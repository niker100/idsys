#!/usr/bin/env python3
"""
Metrics module for evaluating identification systems using idcodes library.

This module provides functions and classes for measuring the performance
of identification systems based on various metrics, such as reliability,
efficiency, error rates, etc.
"""

import numpy as np
import time
import math
from typing import Dict, List, Tuple
from collections import Counter

from .core import IdSystem


class IdMetrics:
    """Class for calculating various metrics for identification systems."""
    
    @staticmethod
    def evaluate_system(
        system: IdSystem, 
        message_set: List[List[int]], 
        num_trials: int = 1000,
        timing_iterations: int = 100,
        p_true_positive: float = 0.5,
        max_messages: int = 10
    ) -> Dict[str, float]:
        """
        Complete evaluation of an identification system.
        
        Args:
            system: The identification system to evaluate
            message_set: List of messages (each message is List[int])
            num_trials: Number of trials for reliability/error rate calculation
            timing_iterations: Number of iterations for timing measurements
            p_true_positive: Probability of a true positive identification
            max_messages: Maximum number of messages to consider for compute intensive metrics
            
        Returns:
            Dictionary containing all metrics
        """
        if not message_set:
            raise ValueError("Message set cannot be empty")
        
        # Get system parameters for code rate calculation
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        system_type = type(encoder).__name__.replace('Encoder', '')
        
        # Calculate message length statistics
        message_lengths = [len(msg) for msg in message_set]
        avg_message_length = np.mean(message_lengths)
        
        # Calculate code rate
        code_rate = IdMetrics._calculate_code_rate(system_type, avg_message_length, gf_exp)
        
        # Calculate reliability and false positive rate
        reliability, fp_rate, false_positives = IdMetrics._calculate_reliability_and_fp_rate(
            system, message_set, num_trials, p_true_positive
        )
        
        # Calculate execution time metrics
        timing_metrics = IdMetrics._calculate_timing_metrics(
            system, message_set, timing_iterations
        )
        
        # Calculate computational efficiency
        comp_efficiency = code_rate / timing_metrics['avg_execution_time_ms'] if timing_metrics['avg_execution_time_ms'] > 0 else 0
        
        # Calculate entropy metrics
        entropy_metrics = IdMetrics._calculate_entropy_metrics(message_set[0:max_messages], gf_exp)
        
        # Calculate tag distribution metrics
        tag_metrics = IdMetrics._calculate_tag_metrics(system, message_set[0:max_messages])
        
        # Compile comprehensive results
        results = {
            # Core performance metrics
            'reliability': reliability,
            'false_positive_rate': fp_rate,
            'false_positives': false_positives,
            'code_rate': code_rate,
            
            # Timing metrics
            'avg_execution_time_ms': timing_metrics['avg_execution_time_ms'],
            'min_execution_time_ms': timing_metrics['min_execution_time_ms'],
            'max_execution_time_ms': timing_metrics['max_execution_time_ms'],
            'std_execution_time_ms': timing_metrics['std_execution_time_ms'],
            
            # Efficiency metrics
            'computational_efficiency': comp_efficiency,
            'throughput_msgs_per_sec': 1000.0 / timing_metrics['avg_execution_time_ms'] if timing_metrics['avg_execution_time_ms'] > 0 else 0,
            
            # Information theory metrics
            'message_entropy': entropy_metrics['message_entropy'],
            'tag_entropy': tag_metrics['tag_entropy'],
            'compression_ratio': entropy_metrics['message_entropy'] / tag_metrics['tag_entropy'] if tag_metrics['tag_entropy'] > 0 else 0,
            
            # System characteristics
            'tag_size_bits': float(gf_exp),
            'avg_message_length': avg_message_length,
            'message_length_std': np.std(message_lengths),
            'unique_tags': tag_metrics['unique_tags'],
            'tag_uniqueness': tag_metrics['tag_uniqueness'],
            'tag_distribution_uniformity': tag_metrics['tag_distribution_uniformity']            
        }
        
        return results
    
    @staticmethod
    def _calculate_code_rate(system_type: str, avg_message_length: float, gf_exp: int) -> float:
        """Calculate effective code rate defined as the ratio of message bits to tag/output bits."""
        # tag size = gf_exp bits
        return avg_message_length * 8 / float(gf_exp)
    
    @staticmethod
    def _calculate_reliability_and_fp_rate(
        system: IdSystem, 
        message_set: List[List[int]], 
        num_trials: int,
        p_true_positive: float = 0.5
    ) -> Tuple[float, float]:
        """Calculate reliability (correct identification rate) and false positive rate."""
        correct = 0
        false_positives = 0
        negatives = 0
        n = len(message_set)
        if n < 2:
            raise ValueError("Message set must contain at least two distinct messages for negative identification scenarios.")

        for _ in range(num_trials):
            # Choose random message and identification scenario
            idx = np.random.randint(0, n)
            msg = message_set[idx]
            is_true = np.random.choice([True, False], p=[p_true_positive, 1-p_true_positive])

            codeword = system.send(msg)
            if is_true:
                # Positive identification scenario
                if system.receive(codeword, msg):
                    correct += 1
            else:
                # Pick a different message than the one sent for negative identification scenario
                other_idx = (idx + np.random.randint(1, n)) % n

                negatives += 1
                if system.receive(codeword, message_set[other_idx]):
                    false_positives += 1
                else:
                    correct += 1


        reliability = correct / num_trials if num_trials > 0 else 0.0
        fp_rate = false_positives / max(negatives, 1)
        return reliability, fp_rate, false_positives
    
    @staticmethod
    def _calculate_timing_metrics(
        system: IdSystem, 
        message_set: List[List[int]], 
        iterations: int
    ) -> Dict[str, float]:
        """Calculate execution time metrics."""
        times = []
        n = len(message_set)
        
        for _ in range(iterations):
            # Choose random message
            message = message_set[np.random.randint(0, n)]
            
            # Time the encoding operation
            start_time = time.perf_counter()

            system.send(message)

            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            times.append(execution_time_ms)
        
        if not times:
            return {
                'avg_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'std_execution_time_ms': 0.0
            }
        
        return {
            'avg_execution_time_ms': float(np.mean(times)),
            'min_execution_time_ms': float(np.min(times)),
            'max_execution_time_ms': float(np.max(times)),
            'std_execution_time_ms': float(np.std(times))
        }
    
    @staticmethod
    def _calculate_entropy_metrics(message_set: List[List[int]], gf_exp: int) -> Dict[str, float]:
        """Calculate entropy metrics for messages."""
        # Flatten all messages and calculate symbol frequency
        all_symbols = []
        for message in message_set:
            all_symbols.extend(message)
        
        if not all_symbols:
            return {'message_entropy': 0.0}
        
        # Calculate symbol frequencies
        symbol_counts = Counter(all_symbols)
        total_symbols = len(all_symbols)
        
        # Calculate entropy
        entropy = 0.0
        for count in symbol_counts.values():
            prob = count / total_symbols
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return {'message_entropy': entropy}
    
    @staticmethod
    def _calculate_tag_metrics(system: IdSystem, sample_messages: List[List[int]]) -> Dict[str, float]:
        """Calculate tag distribution metrics."""
        tags = []
        
        for message in sample_messages:
            tag = system.send(message)
            tags.append(tag)
        
        if not tags:
            return {
                'tag_entropy': 0.0,
                'tag_uniqueness': 0.0,
                'tag_distribution_uniformity': 0.0
            }
        
        # Calculate tag entropy
        tag_counts = Counter(tags)
        total_tags = len(tags)
        tag_entropy = 0.0
        
        for count in tag_counts.values():
            prob = count / total_tags
            if prob > 0:
                tag_entropy -= prob * math.log2(prob)
        
        # Calculate uniqueness (fraction of unique tags)
        unique_tags = len(tag_counts)
        tag_uniqueness = unique_tags / total_tags
        
        # Calculate distribution uniformity (how close to uniform distribution)
        # This is the relative entropy compared to a uniform distribution
        # D(p_X || p_U) = log2(|χ|) - H(X) where X is the random variable, |χ| the size of the alphabet 
        uniformity = math.log2(unique_tags) - tag_entropy if unique_tags > 0 else 0.0
        
        return {
            'unique_tags': unique_tags,
            'tag_entropy': tag_entropy,
            'tag_uniqueness': tag_uniqueness,
            'tag_distribution_uniformity': uniformity
        }
    
    @staticmethod
    def compare_systems(
        systems: Dict[str, IdSystem], 
        message_set: List[List[int]], 
        num_trials: int = 1000,
        timing_iterations: int = 100,
        p_true_positive: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple identification systems.
        
        Args:
            systems: Dictionary mapping system names to IdSystem instances
            message_set: List of messages to test
            num_trials: Number of trials for reliability testing
            timing_iterations: Number of iterations for timing measurements
            
        Returns:
            Dictionary mapping system names to their comprehensive metrics
        """
        results = {}
        
        for name, system in systems.items():
            print(f"Evaluating {name}...")
            results[name] = IdMetrics.evaluate_system(
                system, message_set, num_trials, timing_iterations, p_true_positive
            )
        
        return results