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
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import Counter, defaultdict

from .core import IdSystem, IdEncoder, IdDecoder, generate_test_messages


class IdMetrics:
    """Class for calculating various metrics for identification systems."""
    
    @staticmethod
    def evaluate_system(
        system: IdSystem, 
        message_set: List[List[int]], 
        num_trials: int = 1000,
        timing_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of an identification system.
        
        Args:
            system: The identification system to evaluate
            message_set: List of messages (each message is List[int])
            num_trials: Number of trials for reliability/error rate calculation
            timing_iterations: Number of iterations for timing measurements
            
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
        reliability, fp_rate = IdMetrics._calculate_reliability_and_fp_rate(
            system, message_set, num_trials
        )
        
        # Calculate execution time metrics
        timing_metrics = IdMetrics._calculate_timing_metrics(
            system, message_set, timing_iterations
        )
        
        # Calculate computational efficiency
        comp_efficiency = code_rate / timing_metrics['avg_execution_time_ms'] if timing_metrics['avg_execution_time_ms'] > 0 else 0
        
        # Calculate entropy metrics
        entropy_metrics = IdMetrics._calculate_entropy_metrics(message_set, gf_exp)
        
        # Calculate tag distribution metrics
        tag_metrics = IdMetrics._calculate_tag_metrics(system, message_set[:min(100, len(message_set))])
        
        # Compile comprehensive results
        results = {
            # Core performance metrics
            'reliability': reliability,
            'false_positive_rate': fp_rate,
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
            'tag_uniqueness': tag_metrics['tag_uniqueness'],
            'tag_distribution_uniformity': tag_metrics['tag_distribution_uniformity']            
        }
        
        return results
    
    @staticmethod
    def _calculate_code_rate(system_type: str, avg_message_length: float, gf_exp: int) -> float:
        """Calculate effective code rate based on system type."""
        if system_type in ['SHA1ID']:
            # SHA1 produces 160-bit tags regardless of message length
            return avg_message_length * 8 / 160.0
        elif system_type in ['SHA256ID']:
            # SHA256 produces 256-bit tags regardless of message length
            return avg_message_length * 8 / 256.0
        else:
            # RS/RM systems: tag size = gf_exp bits
            return avg_message_length * 8 / float(gf_exp)
    
    @staticmethod
    def _calculate_reliability_and_fp_rate(
        system: IdSystem, 
        message_set: List[List[int]], 
        num_trials: int
    ) -> Tuple[float, float]:
        """Calculate reliability and false positive rate."""
        correct_count = 0
        false_positives = 0
        total_negative_trials = 0
        
        for _ in range(num_trials):
            # Choose random sender message
            sender_idx = np.random.randint(0, len(message_set))
            sender_message = message_set[sender_idx]
            
            # Choose whether this is a true or false identification scenario
            is_true_id = np.random.choice([True, False])
            
            try:
                codeword = system.send(sender_message)
                
                if is_true_id:
                    # Receiver has the same message
                    receiver_message = sender_message
                    expected_result = True
                    result = system.receive(codeword, receiver_message)
                    if result == expected_result:
                        correct_count += 1
                else:
                    # Receiver has a different message
                    receiver_idx = np.random.randint(0, len(message_set))
                    while receiver_idx == sender_idx and len(message_set) > 1:
                        receiver_idx = np.random.randint(0, len(message_set))
                    receiver_message = message_set[receiver_idx]
                    
                    result = system.receive(codeword, receiver_message)
                    total_negative_trials += 1
                    
                    if result:  # Should be False but got True
                        false_positives += 1
                    else:  # Correctly rejected
                        correct_count += 1
                        
            except Exception:
                # Exception counts as incorrect for reliability, correct rejection for FP
                if not is_true_id:
                    total_negative_trials += 1
                    correct_count += 1  # Exception = correct rejection
        
        reliability = correct_count / num_trials
        fp_rate = false_positives / max(total_negative_trials, 1)
        
        return reliability, fp_rate
    
    @staticmethod
    def _calculate_timing_metrics(
        system: IdSystem, 
        message_set: List[List[int]], 
        iterations: int
    ) -> Dict[str, float]:
        """Calculate execution time metrics."""
        times = []
        
        for _ in range(iterations):
            # Choose random message
            message = message_set[np.random.randint(0, len(message_set))]
            
            # Time the encoding operation
            start_time = time.perf_counter()
            try:
                codeword = system.send(message)
            except Exception:
                continue  # Skip failed encodings
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
            try:
                tag = system.send(message)
                tags.append(tag)
            except Exception:
                continue
        
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
        expected_prob = 1.0 / unique_tags
        uniformity = 0.0
        for count in tag_counts.values():
            actual_prob = count / total_tags
            uniformity += (actual_prob - expected_prob) ** 2
        uniformity = 1.0 - math.sqrt(uniformity / unique_tags)  # Convert to 0-1 scale
        
        return {
            'tag_entropy': tag_entropy,
            'tag_uniqueness': tag_uniqueness,
            'tag_distribution_uniformity': max(0.0, uniformity)
        }
    
    @staticmethod
    def compare_systems(
        systems: Dict[str, IdSystem], 
        message_set: List[List[int]], 
        num_trials: int = 1000,
        timing_iterations: int = 100
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
                system, message_set, num_trials, timing_iterations
            )
        
        return results