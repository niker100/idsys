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
from tqdm import tqdm
from contextlib import nullcontext

from .core import IdSystem, generate_test_messages, create_id_system, generate_structured_messages


def _worker_generate_and_test(args):
    """Worker function for generating messages on-demand and testing, now with metrics calculations."""
    system_type, system_params, codeword, vec_len, gf_exp, batch_size, num_validation_messages, progress_dict, update_frequency, show_progress, message_pattern, worker_seed = args
    
    # Recreate the system in the worker process
    system = create_id_system(system_type, system_params)
    
    false_positives = 0
    times = []
    collided_msgs = []
    
    # For tracking progress
    local_progress = 0
    progress_update_threshold = max(1, min(update_frequency, batch_size // 10)) if show_progress else batch_size
    worker_id = mp.current_process().pid
    
    # Create a generator for this worker's messages
    message_generator = generate_structured_messages(vec_len, message_pattern, gf_exp, batch_size, False, 42, worker_seed)
    
    # For message and tag characteristics
    unique_messages = set()
    first_message = None
    hamming_distances = []
    all_symbols = []
    tag_symbols = []
    
    # Process messages
    messages_processed = 0
    validation_batch = []
    
    for message in message_generator:
        # Store first message as reference for hamming distance
        if first_message is None:
            first_message = list(message)
            
        # Add to unique messages set
        unique_messages.add(tuple(message))
        
        # Update message symbols for PDF calculation
        all_symbols.extend(message)
        
        # Calculate tag for this message for tag PDF
        tag = system.send(message)
        if isinstance(tag, list):
            tag_symbols.extend(tag)
        else:
            tag_symbols.append(tag)
        
        # Calculate hamming distance from reference message
        if first_message:
            hamming_dist = sum(1 for a, b in zip(first_message, message) if a != b)
            hamming_distances.append(hamming_dist)
            
        # Standard message processing for collision testing
        validation_batch.append(message)
        messages_processed += 1
        
        # Process when we have enough messages
        if len(validation_batch) >= num_validation_messages or messages_processed >= batch_size:
            # Time the verification operation
            start_time = time.perf_counter()
            collided_message = system.receive_k(codeword, validation_batch)
            if collided_message:
                false_positives += len(validation_batch)
                collided_msgs.append(collided_message)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            times.append(execution_time_ms)
            
            # Update progress if enabled
            if show_progress:
                local_progress += len(validation_batch)
                if progress_dict is not None and (local_progress >= progress_update_threshold or messages_processed >= batch_size):
                    with progress_dict.get_lock() if hasattr(progress_dict, 'get_lock') else nullcontext():
                        progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
                    local_progress = 0
            
            validation_batch.clear()
        
        if messages_processed >= batch_size:
            break
    
    # Process any remaining messages
    if validation_batch:
        start_time = time.perf_counter()
        collided_message = system.receive_k(codeword, validation_batch)
        if collided_message:
            false_positives += len(validation_batch)
            collided_msgs.append(collided_message)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        times.append(execution_time_ms)
        
        if show_progress and progress_dict is not None:
            local_progress += len(validation_batch)
    
    # Final progress update
    if show_progress and progress_dict is not None and local_progress > 0:
        with progress_dict.get_lock() if hasattr(progress_dict, 'get_lock') else nullcontext():
            progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
    
    # Calculate message and tag PDFs
    message_symbols_total = len(all_symbols)
    message_symbol_counts = Counter(all_symbols)
    message_pdf = {symbol: count/message_symbols_total for symbol, count in message_symbol_counts.items()} if message_symbols_total else {}
    
    tag_symbols_total = len(tag_symbols)
    tag_symbol_counts = Counter(tag_symbols)
    tag_pdf = {symbol: count/tag_symbols_total for symbol, count in tag_symbol_counts.items()} if tag_symbols_total else {}
    
    # Hamming distance statistics
    hamming_stats = {
        'distances': hamming_distances,
        'avg': float(np.mean(hamming_distances)) if hamming_distances else 0.0,
        'min': float(np.min(hamming_distances)) if hamming_distances else 0.0,
        'max': float(np.max(hamming_distances)) if hamming_distances else 0.0,
        'std': float(np.std(hamming_distances)) if hamming_distances else 0.0
    }
    
    # Return all results including metrics
    return (
        false_positives, 
        times, 
        collided_msgs, 
        messages_processed,
        {
            'unique_messages': unique_messages,
            'message_pdf': message_pdf,
            'tag_pdf': tag_pdf,
            'hamming_stats': hamming_stats
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
        save_interval: int = 1000,
        num_processes: int = None,
        show_progress: bool = True,
        message_pattern: str = 'random'
    ) -> Dict[str, float]:
        """
        Complete evaluation of an identification system.
        
        Args:
            system: The identification system to evaluate
            num_messages: Number of messages to generate to evaluate the system
            vec_len: Length of the messages in byte
            num_validation_messages: Number of valid messages at the receiver for k-identification problem
            save_interval: Interval for saving intermediate results
            num_processes: Number of processes to use for parallelization (None for auto)
            show_progress: Whether to show progress bar (default: True)

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
        code_rate = IdMetrics._calculate_code_rate(system_type, message_length, gf_exp)

        # Run parallel message processing with integrated metrics calculation
        fp_rate, false_positives, timing_metrics, collision_metrics, total_messages, aggregated_metrics = IdMetrics._propagate_messages_parallel(
            system, vec_len, num_messages, num_validation_messages, num_processes, show_progress, message_pattern
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
            'num_unique_messages': aggregated_metrics['num_unique_messages'],
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
        """Calculate effective code rate defined as the ratio of message bits to tag/output bits."""
        return avg_message_length * 8 / float(gf_exp)
    
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
        show_progress: bool = True,
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
        
        # Create progress manager
        progress_dict = None
        manager = None
        if show_progress:
            manager = mp.Manager()
            progress_dict = manager.dict()
        
        update_frequency = max(100, remaining_messages // 100) if show_progress else remaining_messages
        
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
                progress_dict,
                update_frequency,
                show_progress,
                message_pattern,
                worker_seed
            ))
        
        # Initialize result aggregation
        total_false_positives = 0
        all_times = []
        collided_msgs = []
        total_messages_processed = 1  # Start with 1 for the first message used for codeword
        
        # For aggregating metrics
        all_unique_messages = set()
        all_message_pdfs = []
        all_tag_pdfs = []
        all_hamming_distances = []
        
        if num_processes == 1 or len(worker_args) == 1:
            # Single process execution
            for args in worker_args:
                fp, times, collided, processed, metrics = _worker_generate_and_test(args)
                total_false_positives += fp
                all_times.extend(times)
                collided_msgs.extend(collided)
                total_messages_processed += processed
                
                # Aggregate metrics
                all_unique_messages.update(metrics['unique_messages'])
                all_message_pdfs.append(metrics['message_pdf'])
                all_tag_pdfs.append(metrics['tag_pdf'])
                all_hamming_distances.extend(metrics['hamming_stats']['distances'])
        else:
            # Multi-process execution
            with mp.Pool(processes=num_processes) as pool:
                if show_progress:
                    jobs = [pool.apply_async(_worker_generate_and_test, (args,)) for args in worker_args]
                    
                    desc = f"Processing {num_messages} messages ({num_processes} proc)"
                    sleep_time = 0.1 if num_messages < 10000 else 0.5
                    
                    with tqdm(total=num_messages-1, desc=desc) as pbar:
                        last_total = 0
                        
                        while any(not job.ready() for job in jobs):
                            current_total = sum(progress_dict.values()) if progress_dict else 0
                            if current_total > last_total:
                                pbar.update(current_total - last_total)
                                last_total = current_total
                            time.sleep(sleep_time)
                        
                        for job in jobs:
                            fp, times, collided, processed, metrics = job.get()
                            total_false_positives += fp
                            all_times.extend(times)
                            collided_msgs.extend(collided)
                            total_messages_processed += processed
                            
                            # Aggregate metrics
                            all_unique_messages.update(metrics['unique_messages'])
                            all_message_pdfs.append(metrics['message_pdf'])
                            all_tag_pdfs.append(metrics['tag_pdf'])
                            all_hamming_distances.extend(metrics['hamming_stats']['distances'])
                        
                        current_total = sum(progress_dict.values()) if progress_dict else 0
                        if current_total > last_total:
                            pbar.update(current_total - last_total)
                else:
                    results = pool.map(_worker_generate_and_test, worker_args)
                    for fp, times, collided, processed, metrics in results:
                        total_false_positives += fp
                        all_times.extend(times)
                        collided_msgs.extend(collided)
                        total_messages_processed += processed
                        
                        # Aggregate metrics
                        all_unique_messages.update(metrics['unique_messages'])
                        all_message_pdfs.append(metrics['message_pdf'])
                        all_tag_pdfs.append(metrics['tag_pdf'])
                        all_hamming_distances.extend(metrics['hamming_stats']['distances'])
            
            # Clean up manager
            if manager is not None:
                manager.shutdown()
        
        if not all_times:
            return 0.0, total_false_positives, {
                'avg_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'std_execution_time_ms': 0.0
            }, {
                'avg_hamming_distance': 0.0,
                'min_hamming_distance': 0.0,
                'max_hamming_distance': 0.0,
                'std_hamming_distance': 0.0
            }, total_messages_processed, {
                'num_unique_messages': 0,
                'message_pdf': {},
                'tag_pdf': {}
            }
        
        # Calculate false positive rate
        fp_rate = total_false_positives / max(1, total_messages_processed - 1)

        # Calculate collision hamming distances
        collision_hamming_distances = []
        if collided_msgs:
            for msg in collided_msgs:
                hamming_distance = sum(1 for a, b in zip(first_message, msg) if a != b)
                collision_hamming_distances.append(hamming_distance)

        # Aggregate collision metrics
        collision_metrics = {
            'avg_hamming_distance': float(np.mean(collision_hamming_distances)) if collision_hamming_distances else 0.0,
            'min_hamming_distance': float(np.min(collision_hamming_distances)) if collision_hamming_distances else 0.0,
            'max_hamming_distance': float(np.max(collision_hamming_distances)) if collision_hamming_distances else 0.0,
            'std_hamming_distance': float(np.std(collision_hamming_distances)) if collision_hamming_distances else 0.0
        }
        
        # Aggregate timing metrics
        timing_metrics = {
            'avg_execution_time_ms': float(np.mean(all_times)),
            'min_execution_time_ms': float(np.min(all_times)),
            'max_execution_time_ms': float(np.max(all_times)),
            'std_execution_time_ms': float(np.std(all_times))
        }
        
        # Aggregate message set metrics
        message_hamming_metrics = {
            'avg_hamming_distance': float(np.mean(all_hamming_distances)) if all_hamming_distances else 0.0,
            'min_hamming_distance': float(np.min(all_hamming_distances)) if all_hamming_distances else 0.0,
            'max_hamming_distance': float(np.max(all_hamming_distances)) if all_hamming_distances else 0.0,
            'std_hamming_distance': float(np.std(all_hamming_distances)) if all_hamming_distances else 0.0
        }
        
        # Combine PDFs 
        combined_message_pdf = IdMetrics._merge_pdfs(all_message_pdfs)
        combined_tag_pdf = IdMetrics._merge_pdfs(all_tag_pdfs)
        
        # Compile aggregated metrics
        aggregated_metrics = {
            'num_unique_messages': len(all_unique_messages),
            'message_pdf': combined_message_pdf,
            'tag_pdf': combined_tag_pdf,
            'hamming_metrics': message_hamming_metrics
        }
        
        return fp_rate, total_false_positives, timing_metrics, collision_metrics, total_messages_processed, aggregated_metrics
    
    @staticmethod
    def _calculate_symbol_pdf(message_set) -> Dict[str, float]:
        """Calculate entropy metrics for messages or tags."""
        # Support both List[List[int]] and List[int]
        all_symbols = []
        if not message_set:
            return {}
        if isinstance(message_set[0], (list, tuple)):
            # Flatten all messages
            for message in message_set:
                all_symbols.extend(message)
        else:
            # Already a flat list of symbols
            all_symbols = list(message_set)
        symbol_counts = Counter(all_symbols)
        total_symbols = len(all_symbols)

        return {symbol: count / total_symbols for symbol, count in symbol_counts.items()} if total_symbols > 0 else {}

    
    @staticmethod
    def compare_systems(
        systems: Dict[str, IdSystem], 
        num_messages: int = 1000, 
        vec_len: int = 16,
        message_subset_size: int = 10,
        num_processes: int = None,
        message_pattern: str = 'random',
        show_progress: bool = True
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
                num_processes=num_processes, show_progress=show_progress,
                message_pattern=message_pattern
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