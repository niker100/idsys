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
import multiprocessing as mp
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from .core import IdSystem, generate_test_messages, create_id_system, generate_structured_messages


def _worker_generate_and_test(args):
    """Memory-optimized worker function for generating and testing messages."""
    system_type, system_params, codeword, vec_len, gf_exp, batch_size, num_validation_messages, message_pattern, worker_seed = args
    
    # Recreate the system in the worker process
    system = create_id_system(system_type, system_params)
    
    false_positives = 0
    execution_time_stats = {
        'total': 0.0,
        'count': 0,
        'min': float('inf'),
        'max': float('-inf'),
        'sum_squares': 0.0  # For computing variance later
    }
    
    # Sample collided messages (limit to reasonable number)
    max_collided_samples = min(100, batch_size // 100)
    collided_msgs_sample = []
    
    # Create a generator for this worker's messages
    message_generator = generate_structured_messages(vec_len, message_pattern, gf_exp, batch_size, False, 42, worker_seed)
    
    first_message = None
    
    # For online calculation of hamming distance statistics (Welford's algorithm)
    hamming_stats = {
        'count': 0,
        'mean': 0.0,
        'M2': 0.0,  # For online variance calculation
        'min': float('inf'),
        'max': float('-inf')
    }
    
    # Use counters for PDFs instead of storing all symbols
    message_symbol_counts = Counter()
    tag_symbol_counts = Counter()
    
    # Process messages
    messages_processed = 0
    validation_batch = []
    
    for message in message_generator:
        # Store first message as reference for hamming distance
        if first_message is None:
            first_message = list(message)
        
        # Update message symbols for PDF calculation
        message_symbol_counts.update(message)
        
        # Calculate tag for this message and update tag PDF
        tag = system.send(message)
        if isinstance(tag, list):
            tag_symbol_counts.update(tag)
        else:
            tag_symbol_counts[tag] += 1
        
        # Calculate hamming distance using Welford's online algorithm
        if first_message:
            hamming_dist = sum(1 for a, b in zip(first_message, message) if a != b)
            
            hamming_stats['count'] += 1
            delta = hamming_dist - hamming_stats['mean']
            hamming_stats['mean'] += delta / hamming_stats['count']
            delta2 = hamming_dist - hamming_stats['mean']
            hamming_stats['M2'] += delta * delta2
            
            hamming_stats['min'] = min(hamming_stats['min'], hamming_dist)
            hamming_stats['max'] = max(hamming_stats['max'], hamming_dist)
        
        # Add to validation batch
        validation_batch.append(message)
        messages_processed += 1
        
        # Process when we have enough messages
        if len(validation_batch) >= num_validation_messages or messages_processed >= batch_size:
            # Time the verification operation
            start_time = time.perf_counter()
            collided_message = system.receive_k(codeword, validation_batch)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            # Update timing statistics
            execution_time_stats['total'] += execution_time_ms
            execution_time_stats['count'] += 1
            execution_time_stats['min'] = min(execution_time_stats['min'], execution_time_ms)
            execution_time_stats['max'] = max(execution_time_stats['max'], execution_time_ms)
            execution_time_stats['sum_squares'] += execution_time_ms ** 2
            
            if collided_message:
                false_positives += len(validation_batch)
                # Keep only a limited sample of collided messages
                if len(collided_msgs_sample) < max_collided_samples:
                    collided_msgs_sample.append(collided_message)
            
            validation_batch.clear()
        
        if messages_processed >= batch_size:
            break
    
    # Process any remaining messages
    if validation_batch:
        # Similar processing as above
        start_time = time.perf_counter()
        collided_message = system.receive_k(codeword, validation_batch)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        execution_time_stats['total'] += execution_time_ms
        execution_time_stats['count'] += 1
        execution_time_stats['min'] = min(execution_time_stats['min'], execution_time_ms)
        execution_time_stats['max'] = max(execution_time_stats['max'], execution_time_ms)
        execution_time_stats['sum_squares'] += execution_time_ms ** 2
        
        if collided_message:
            false_positives += len(validation_batch)
            if len(collided_msgs_sample) < max_collided_samples:
                collided_msgs_sample.append(collided_message)
    
    # Calculate PDFs from counts
    message_symbols_total = sum(message_symbol_counts.values())
    message_pdf = {symbol: count/message_symbols_total for symbol, count in message_symbol_counts.items()} if message_symbols_total else {}
    
    tag_symbols_total = sum(tag_symbol_counts.values())
    tag_pdf = {symbol: count/tag_symbols_total for symbol, count in tag_symbol_counts.items()} if tag_symbols_total else {}
    
    # Calculate std dev for hamming distances
    hamming_std = math.sqrt(hamming_stats['M2'] / hamming_stats['count']) if hamming_stats['count'] > 1 else 0.0
    
    # Calculate execution time statistics
    mean_time = execution_time_stats['total'] / execution_time_stats['count'] if execution_time_stats['count'] > 0 else 0.0
    time_variance = (execution_time_stats['sum_squares'] / execution_time_stats['count'] - mean_time**2) if execution_time_stats['count'] > 1 else 0.0
    time_std = math.sqrt(max(0, time_variance))
    
    # Return pre-aggregated results instead of raw data
    return (
        false_positives,
        {
            'mean': mean_time,
            'min': execution_time_stats['min'] if execution_time_stats['count'] > 0 else 0.0,
            'max': execution_time_stats['max'] if execution_time_stats['count'] > 0 else 0.0,
            'std': time_std,
            'count': execution_time_stats['count']
        },
        collided_msgs_sample,
        messages_processed,
        {
            'message_pdf': message_pdf,
            'tag_pdf': tag_pdf,
            'hamming_stats': {
                'avg': hamming_stats['mean'],
                'min': hamming_stats['min'] if hamming_stats['count'] > 0 else 0.0,
                'max': hamming_stats['max'] if hamming_stats['count'] > 0 else 0.0,
                'std': hamming_std,
                'count': hamming_stats['count']
            }
        }
    )

class IdMetrics:
    """Class for calculating various metrics for identification systems."""
        
    @staticmethod
    def _merge_pdfs(pdfs_list):
        """Merge multiple probability density functions into a single combined PDF."""
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
    def evaluate_system(
        system: IdSystem,
        num_messages: int = 100000,
        vec_len: int = 16,
        num_validation_messages: int = 1,
        num_processes: int = None,
        message_pattern: str = 'random'
    ) -> Dict[str, float]:
        """
        Complete evaluation of an identification system.
        
        Args:
            system: The identification system to evaluate
            num_messages: Number of messages to generate to evaluate the system
            vec_len: Length of the messages in byte
            num_validation_messages: Number of valid messages at the receiver for k-identification problem
            num_processes: Number of processes to use for parallelization (None for auto)

        Returns:
            Dictionary containing all metrics
        """
        
        # Get system parameters for code rate calculation
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        system_type = type(encoder).__name__.replace('Encoder', '')

        # Calculate approximate message length for code rate
        if gf_exp >= 33:
            message_length = vec_len // 8
        elif gf_exp >= 17:
            message_length = vec_len // 4
        elif gf_exp >= 9:
            message_length = vec_len // 2
        else:
            message_length = vec_len
        
        # Calculate code rate
        code_rate = IdMetrics._calculate_code_rate(system_type, vec_len*8 , gf_exp)

        # Run parallel message processing with integrated metrics calculation
        fp_rate, false_positives, timing_metrics, collision_metrics, total_messages, aggregated_metrics = IdMetrics._propagate_messages_parallel(
            system, vec_len, num_messages, num_validation_messages, num_processes, message_pattern
        )

        # Compile comprehensive results
        results = {
            # Core performance metrics
            'total_messages': total_messages,
            'false_positive_rate': fp_rate,
            'false_positives': false_positives,
            'code_rate': code_rate,
            
            # Timing metrics
            'avg_execution_time_ms': timing_metrics['avg_execution_time_ms'],
            'min_execution_time_ms': timing_metrics['min_execution_time_ms'],
            'max_execution_time_ms': timing_metrics['max_execution_time_ms'],
            'std_execution_time_ms': timing_metrics['std_execution_time_ms'],
            
            # Efficiency metrics
            'throughput_msgs_per_sec': 1000.0 / timing_metrics['avg_execution_time_ms'] if timing_metrics['avg_execution_time_ms'] > 0 else 0,
            

            # Message set characteristics            
            'message_pdf': aggregated_metrics['message_pdf'],
            'avg_hamming_distance': aggregated_metrics['hamming_metrics']['avg_hamming_distance'],
            'min_hamming_distance': aggregated_metrics['hamming_metrics']['min_hamming_distance'],
            'max_hamming_distance': aggregated_metrics['hamming_metrics']['max_hamming_distance'],
            'std_hamming_distance': aggregated_metrics['hamming_metrics']['std_hamming_distance'],
            'message_length': message_length,

            # Collision metrics
            'collisions_avg_hamming_distance': collision_metrics['avg_hamming_distance'],
            'collisions_min_hamming_distance': collision_metrics['min_hamming_distance'],
            'collisions_max_hamming_distance': collision_metrics['max_hamming_distance'],
            'collisions_std_hamming_distance': collision_metrics['std_hamming_distance'],
            
            # Tag characteristics
            'tag_size_bits': float(gf_exp),
            'tag_pdf': aggregated_metrics['tag_pdf'],
        }
        
        return results
    
    @staticmethod
    def _calculate_code_rate(system_type: str, avg_message_length: float, gf_exp: int) -> float:
        """Calculate effective code rate defined as the ratio of log2(log2(N))/output bits."""
        return np.log2(np.log2(avg_message_length)) / float(2**gf_exp)
    
    @staticmethod
    def _get_system_info(system: IdSystem) -> Tuple[str, Dict]:
        """Extract system type and parameters for recreation."""
        encoder = system.encoder
        system_type = type(encoder).__name__.replace('Encoder', '')
        params = getattr(encoder, 'parameters', {})
        return system_type, params
    
    @staticmethod
    def _propagate_messages_parallel(
        system: IdSystem,
        vec_len: int,
        num_messages: int,
        num_validation_messages: int = 1,
        num_processes: int = None,
        message_pattern: str = 'random'
    ) -> Tuple[float, int, Dict[str, float], Dict[str, float], int, Dict]:
        """Memory-optimized parallelized version that generates messages and calculates metrics on demand."""
        if num_messages < 2:
            raise ValueError("Need at least two messages for meaningful evaluation.")
        
        # Generate only the first message
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        
        # Generate just one message for the reference
        first_message_gen = generate_structured_messages(vec_len, message_pattern, gf_exp, 1, True)
        first_message = next(first_message_gen)
        
        # Send the first message to get the codeword
        codeword = system.send(first_message)
        
        # Set number of processes
        if num_processes is None:
            num_processes = min(mp.cpu_count(), (num_messages - 1) // 1000 + 1)
        
        # Calculate batch size per worker process
        remaining_messages = num_messages - 1
        batch_size_per_process = remaining_messages // num_processes
        
        # Get system info for recreation in worker processes
        system_type, system_params = IdMetrics._get_system_info(system)
        
        # Prepare worker arguments
        worker_args = []
        for i in range(num_processes):
            actual_batch_size = batch_size_per_process + (remaining_messages % num_processes if i == num_processes-1 else 0)
            worker_seed = i
            
            worker_args.append((
                system_type, 
                system_params, 
                codeword, 
                vec_len, 
                gf_exp, 
                actual_batch_size, 
                num_validation_messages,
                message_pattern,
                worker_seed
            ))
        
        # Initialize result aggregation
        total_false_positives = 0
        total_messages_processed = 1  # Start with 1 for the first message used for codeword
        
        # For aggregating metrics
        all_message_pdfs = []
        all_tag_pdfs = []
        
        # For aggregating hamming statistics with Welford's parallel algorithm
        combined_hamming_stats = {
            'count': 0,
            'mean': 0.0,
            'M2': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }
        
        # For aggregating timing statistics
        combined_timing_stats = {
            'weighted_sum': 0.0,
            'count': 0,
            'combined_variance': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }
        
        # For tracking collisions
        all_collided_msgs = []
        max_collision_samples = 1000  # Limit total stored collision samples        

        # Multi-process execution
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(_worker_generate_and_test, worker_args)        
        
        # Process results
        for fp, times_stats, collided, processed, metrics in results:
            total_false_positives += fp
            total_messages_processed += processed
            
            # Collect PDFs for later merging
            all_message_pdfs.append(metrics['message_pdf'])
            all_tag_pdfs.append(metrics['tag_pdf'])
            
            # Store limited sample of collided messages
            remaining_slots = max_collision_samples - len(all_collided_msgs)
            if remaining_slots > 0:
                all_collided_msgs.extend(collided[:min(len(collided), remaining_slots)])
            
            # Aggregate timing statistics with weighted mean
            if times_stats['count'] > 0:
                combined_timing_stats['weighted_sum'] += times_stats['mean'] * times_stats['count']
                combined_timing_stats['count'] += times_stats['count']
                combined_timing_stats['min'] = min(combined_timing_stats['min'], times_stats['min'])
                combined_timing_stats['max'] = max(combined_timing_stats['max'], times_stats['max'])
                
                # Combine variances using parallel algorithm
                if times_stats['count'] > 1:
                    combined_timing_stats['combined_variance'] += (times_stats['std'] ** 2) * (times_stats['count'] - 1)
            
            # Combine hamming distance statistics using parallel algorithm
            hamming_worker = metrics['hamming_stats']
            if hamming_worker['count'] > 0:
                n1 = combined_hamming_stats['count']
                n2 = hamming_worker['count']
                
                if n1 == 0:
                    # First batch
                    combined_hamming_stats['mean'] = hamming_worker['avg']
                    combined_hamming_stats['min'] = hamming_worker['min']
                    combined_hamming_stats['max'] = hamming_worker['max']
                    combined_hamming_stats['M2'] = (hamming_worker['std'] ** 2) * (n2 - 1) if n2 > 1 else 0
                    combined_hamming_stats['count'] = n2
                else:
                    # Combine with existing stats
                    delta = hamming_worker['avg'] - combined_hamming_stats['mean']
                    combined_mean = (n1 * combined_hamming_stats['mean'] + n2 * hamming_worker['avg']) / (n1 + n2)
                    
                    # Combine M2 values for variance
                    combined_hamming_stats['M2'] += (hamming_worker['std'] ** 2) * (n2 - 1) + \
                                                delta**2 * n1 * n2 / (n1 + n2)
                    
                    combined_hamming_stats['mean'] = combined_mean
                    combined_hamming_stats['min'] = min(combined_hamming_stats['min'], hamming_worker['min'])
                    combined_hamming_stats['max'] = max(combined_hamming_stats['max'], hamming_worker['max'])
                    combined_hamming_stats['count'] += n2
        
        # Calculate final metrics
        
        # Timing metrics
        avg_execution_time = (combined_timing_stats['weighted_sum'] / combined_timing_stats['count'] 
                            if combined_timing_stats['count'] > 0 else 0.0)
        
        # Calculate final std dev for timing
        if combined_timing_stats['count'] > 1:
            timing_variance = combined_timing_stats['combined_variance'] / (combined_timing_stats['count'] - 1)
            timing_std = math.sqrt(timing_variance)
        else:
            timing_std = 0.0
        
        timing_metrics = {
            'avg_execution_time_ms': avg_execution_time,
            'min_execution_time_ms': combined_timing_stats['min'] if combined_timing_stats['count'] > 0 else 0.0,
            'max_execution_time_ms': combined_timing_stats['max'] if combined_timing_stats['count'] > 0 else 0.0,
            'std_execution_time_ms': timing_std
        }
        
        # Calculate final std dev for hamming distances
        hamming_std = math.sqrt(combined_hamming_stats['M2'] / combined_hamming_stats['count']) if combined_hamming_stats['count'] > 1 else 0.0
        
        message_hamming_metrics = {
            'avg_hamming_distance': combined_hamming_stats['mean'],
            'min_hamming_distance': combined_hamming_stats['min'] if combined_hamming_stats['count'] > 0 else 0.0,
            'max_hamming_distance': combined_hamming_stats['max'] if combined_hamming_stats['count'] > 0 else 0.0,
            'std_hamming_distance': hamming_std
        }
        
        # Calculate false positive rate
        fp_rate = total_false_positives / max(1, total_messages_processed - 1)
        
        # Calculate collision metrics - only on the sampled collisions
        collision_metrics = {
            'avg_hamming_distance': 0.0,
            'min_hamming_distance': 0.0,
            'max_hamming_distance': 0.0,
            'std_hamming_distance': 0.0
        }
        
        if all_collided_msgs and first_message:
            collision_hamming_distances = [sum(1 for a, b in zip(first_message, msg) if a != b) for msg in all_collided_msgs]
            if collision_hamming_distances:
                collision_metrics = {
                    'avg_hamming_distance': float(np.mean(collision_hamming_distances)),
                    'min_hamming_distance': float(np.min(collision_hamming_distances)),
                    'max_hamming_distance': float(np.max(collision_hamming_distances)),
                    'std_hamming_distance': float(np.std(collision_hamming_distances))
                }
        
        # Combine PDFs efficiently
        combined_message_pdf = IdMetrics._merge_pdfs(all_message_pdfs)
        combined_tag_pdf = IdMetrics._merge_pdfs(all_tag_pdfs)
        
        # Compile aggregated metrics
        aggregated_metrics = {
            'message_pdf': combined_message_pdf,
            'tag_pdf': combined_tag_pdf,
            'hamming_metrics': message_hamming_metrics
        }
        
        return fp_rate, total_false_positives, timing_metrics, collision_metrics, total_messages_processed, aggregated_metrics    

    
    @staticmethod
    def compare_systems(
        systems: Dict[str, IdSystem], 
        num_messages: int = 1000, 
        vec_len: int = 16,
        message_subset_size: int = 10,
        num_processes: int = None,
        message_pattern: str = 'random'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple identification systems.
        
        Args:
            systems: Dictionary mapping system names to IdSystem instances
            num_messages: Number of messages to test
            vec_len: Vector length in bytes
            message_subset_size: Size of message subset for compute intensive metrics
            num_processes: Number of processes to use for parallelization
            
        Returns:
            Dictionary mapping system names to their comprehensive metrics
        """
        results = {}
        
        for name, system in systems.items():
            print(f"Evaluating {name}...")
            results[name] = IdMetrics.evaluate_system(
                system, num_messages, vec_len, message_subset_size=message_subset_size,
                num_processes=num_processes, message_pattern=message_pattern
            )
        
        return results
    


    @DeprecationWarning
    @staticmethod
    def evaluate_system_old(
        system: IdSystem,
        message_set: List[List[int]],
        num_trials: int = 1000,
        p_true_positive: float = 0.5,
        timing_iterations: int = 1000,
        message_subset_size: int = 10
    ) -> Dict[str, float]:
        """
        Complete evaluation of an identification system.

        Returns:
            Dictionary containing all metrics
        """
        
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
        entropy_metrics = IdMetrics._calculate_entropy_metrics(message_set[0:message_subset_size], gf_exp)
        
        # Calculate tag distribution metrics
        tag_metrics = IdMetrics._calculate_tag_metrics(system, message_set[0:message_subset_size])
        
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
            'tag_distribution_uniformity': tag_metrics['tag_distribution_uniformity'],
            'tag_max_value': tag_metrics['tag_max_value'],        
        }
        
        return results
    
    @DeprecationWarning
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
    
    @DeprecationWarning
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