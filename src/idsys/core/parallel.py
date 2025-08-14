"""
Parallel processing functionality for identification system metrics.

This module provides worker functions and utilities for parallelizing
the evaluation of identification systems.
"""

import time
import math
import multiprocessing as mp
from collections import Counter
import numpy as np

from .idsystems import create_id_system, IdSystem
from .message_generation import generate_structured_messages

def worker_generate_and_test(args: tuple) -> tuple:
    """
    Memory-optimized worker function for generating and testing messages.
    
    This function is designed to be used with multiprocessing to parallelize
    identification system evaluation. It generates messages, tests them against
    a codeword, and collects metrics.
    
    Args:
        args: Tuple containing:
            - system_type: Type of identification system
            - system_params: Parameters for the system
            - codeword: Codeword to test messages against
            - vec_len: Vector length for messages
            - gf_exp: Galois field exponent
            - batch_size: Number of messages to generate
            - num_validation_messages: Number of messages to validate at once
            - message_pattern: Pattern for message generation
            - worker_seed: Seed for this worker
            - calculate_pdfs: Whether to calculate PDFs
            
    Returns:
        Tuple containing:
            - false_positives: Number of false positive matches
            - execution_time_stats: Dictionary of timing statistics
            - collided_msgs_sample: Sample of collided messages
            - messages_processed: Total messages processed
            - metrics_data: Dictionary of additional metrics
    """
    system_type, system_params, codeword, vec_len, gf_exp, batch_size, num_validation_messages, message_pattern, worker_seed, calculate_pdfs = args
    
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
    max_collided_samples = 100
    collided_msgs_sample = []
    
    # Create a generator for this worker's messages
    message_generator = generate_structured_messages(
        vec_len=vec_len, 
        pattern_type=message_pattern, 
        gf_exp=gf_exp, 
        target_count=batch_size, 
        generate_first=False, 
        seed=42, 
        worker_offset=worker_seed
    )
    
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
    if calculate_pdfs:
        message_symbol_counts = Counter()
        tag_symbol_counts = Counter()
    else:
        message_symbol_counts = None
        tag_symbol_counts = None
    
    # Process messages
    messages_processed = 0
    validation_batch = []
    
    for message in message_generator:
        # Store first message as reference for hamming distance
        if first_message is None:
            first_message = list(message)
        
        # Update message symbols for PDF calculation
        if calculate_pdfs:
            message_symbol_counts.update(message)
        
        # Calculate tag for this message and update tag PDF
        tag = system.send(message)
        if calculate_pdfs:
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
    if calculate_pdfs and message_symbol_counts is not None:
        message_symbols_total = sum(message_symbol_counts.values())
        message_pdf = {symbol: count/message_symbols_total 
                      for symbol, count in message_symbol_counts.items()} if message_symbols_total else {}
        
        tag_symbols_total = sum(tag_symbol_counts.values())
        tag_pdf = {symbol: count/tag_symbols_total 
                  for symbol, count in tag_symbol_counts.items()} if tag_symbols_total else {}
    else:
        message_pdf = {}
        tag_pdf = {}
    
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

def propagate_messages_parallel(
    system: IdSystem,
    vec_len: int,
    num_messages: int,
    num_validation_messages: int = 1,
    num_processes: int = None,
    message_pattern: str = 'random',
    calculate_pdfs: bool = True
) -> tuple:
    """
    Process messages in parallel to evaluate identification system metrics.
    
    This function distributes message generation and testing across multiple
    processes for efficient system evaluation.
    
    Args:
        system: Identification system to evaluate
        vec_len: Vector length for messages
        num_messages: Total number of messages to generate
        num_validation_messages: Number of messages to validate at once
        num_processes: Number of parallel processes to use (None for auto)
        message_pattern: Pattern for message generation
        calculate_pdfs: Whether to calculate probability density functions
        
    Returns:
        Tuple containing:
            - fp_rate: False positive rate
            - total_false_positives: Total number of false positives
            - timing_metrics: Dictionary of timing metrics
            - collision_metrics: Dictionary of collision metrics
            - total_messages_processed: Total messages processed
            - aggregated_metrics: Dictionary of additional metrics
    """
    from .statistics import StatisticsUtils
    
    if num_messages < 2:
        raise ValueError("Need at least two messages for meaningful evaluation.")
    
    # Get system parameters
    encoder = system.encoder
    params = getattr(encoder, 'parameters', {})
    gf_exp = params.get('gf_exp', 8)
    
    # Generate just one message for the reference
    first_message_gen = generate_structured_messages(
        vec_len=vec_len, 
        pattern_type=message_pattern, 
        gf_exp=gf_exp, 
        target_count=1, 
        generate_first=True
    )
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
    system_type = type(encoder).__name__.replace('Encoder', '')
    system_params = params
    
    # Prepare worker arguments
    worker_args = []
    for i in range(num_processes):
        actual_batch_size = batch_size_per_process
        if i == num_processes - 1:
            # Add remaining messages to last process
            actual_batch_size += remaining_messages % num_processes
            
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
            worker_seed,
            calculate_pdfs
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
        results = pool.map(worker_generate_and_test, worker_args)        
    
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
        collision_hamming_distances = [sum(1 for a, b in zip(first_message, msg) if a != b) 
                                     for msg in all_collided_msgs]
        if collision_hamming_distances:
            collision_metrics = {
                'avg_hamming_distance': float(np.mean(collision_hamming_distances)),
                'min_hamming_distance': float(np.min(collision_hamming_distances)),
                'max_hamming_distance': float(np.max(collision_hamming_distances)),
                'std_hamming_distance': float(np.std(collision_hamming_distances))
            }
    
    # Combine PDFs efficiently
    combined_message_pdf = StatisticsUtils.merge_pdfs(all_message_pdfs) if calculate_pdfs else {}
    combined_tag_pdf = StatisticsUtils.merge_pdfs(all_tag_pdfs) if calculate_pdfs else {}
    
    # Compile aggregated metrics
    aggregated_metrics = {
        'message_pdf': combined_message_pdf,
        'tag_pdf': combined_tag_pdf,
        'hamming_metrics': message_hamming_metrics
    }
    
    return fp_rate, total_false_positives, timing_metrics, collision_metrics, total_messages_processed, aggregated_metrics