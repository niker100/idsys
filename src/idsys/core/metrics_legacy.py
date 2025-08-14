"""
Legacy metrics functionality for backward compatibility.

This module contains deprecated metrics functionality that is 
maintained for backward compatibility. New code should use the
functions in the metrics module instead.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import warnings

from .idsystems import IdSystem

def calculate_reliability_and_fp_rate(
    system: IdSystem, 
    message_set: List[List[int]], 
    num_trials: int,
    p_true_positive: float = 0.5
) -> Tuple[float, float, int]:
    """
    Calculate reliability and false positive rate (deprecated).
    
    Args:
        system: Identification system to evaluate
        message_set: List of messages for evaluation
        num_trials: Number of trials to run
        p_true_positive: Probability of selecting a true positive scenario
        
    Returns:
        Tuple of (reliability, false_positive_rate, false_positives)
        
    Deprecated:
        Use IdMetrics.evaluate_system instead.
    """
    warnings.warn(
        "calculate_reliability_and_fp_rate is deprecated. Use IdMetrics.evaluate_system instead.",
        DeprecationWarning, stacklevel=2
    )
    
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

def calculate_timing_metrics(
    system: IdSystem, 
    message_set: List[List[int]], 
    iterations: int
) -> Dict[str, float]:
    """
    Calculate execution time metrics (deprecated).
    
    Args:
        system: Identification system to evaluate
        message_set: List of messages for evaluation
        iterations: Number of iterations to run
        
    Returns:
        Dictionary of timing metrics
        
    Deprecated:
        Use IdMetrics.evaluate_system instead.
    """
    warnings.warn(
        "calculate_timing_metrics is deprecated. Use IdMetrics.evaluate_system instead.",
        DeprecationWarning, stacklevel=2
    )
    
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