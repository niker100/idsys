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
from collections import Counter
from tqdm import tqdm
from contextlib import nullcontext

from .core import IdSystem, generate_test_messages, create_id_system, generate_structured_messages


def _worker_generate_and_test(args):
    """Worker function for generating messages on-demand and testing."""
    system_type, system_params, codeword, vec_len, gf_exp, batch_size, num_validation_messages, progress_dict, update_frequency, show_progress, message_pattern, worker_seed = args
    
    # Recreate the system in the worker process
    system = create_id_system(system_type, system_params)
    
    false_positives = 0
    times = []
    collided_msgs = []
    
    # For tracking progress with minimal updates (only if progress bar is enabled)
    local_progress = 0
    progress_update_threshold = max(1, min(update_frequency, batch_size // 10)) if show_progress else batch_size
    worker_id = mp.current_process().pid
    
    # Create a generator for this worker's messages with unique seed
    # Each worker gets a different starting point to avoid duplicates
    message_generator = generate_structured_messages(vec_len, message_pattern, gf_exp, batch_size, False, 42, worker_seed)
    
    # # Skip ahead based on worker seed to ensure different messages per worker
    # if worker_seed > 0:
    #     for _ in range(worker_seed):
    #         try:
    #             next(message_generator)
    #         except StopIteration:
    #             break
    
    # Process messages as they're generated
    messages_processed = 0
    validation_batch = []
    
    for message in message_generator:            
        validation_batch.append(message)
        messages_processed += 1
        
        # Process when we have enough messages or reached the end
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
            
            # Update progress if enabled - FIXED: Use actual batch size processed
            if show_progress:
                local_progress += len(validation_batch)
                
                # Update shared progress counter infrequently
                if progress_dict is not None and (local_progress >= progress_update_threshold or messages_processed >= batch_size):
                    with progress_dict.get_lock() if hasattr(progress_dict, 'get_lock') else nullcontext():
                        progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
                    local_progress = 0
            
            # Clear the batch to free memory
            validation_batch.clear()
        
        # Break if we've processed enough messages
        if messages_processed >= batch_size:
            break
    
    # Process any remaining messages in the batch
    if validation_batch:
        start_time = time.perf_counter()
        collided_message = system.receive_k(codeword, validation_batch)
        if collided_message:
            false_positives += len(validation_batch)
            collided_msgs.append(collided_message)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        times.append(execution_time_ms)
        
        # Update progress for remaining messages
        if show_progress and progress_dict is not None:
            local_progress += len(validation_batch)
    
    # Ensure any remaining progress is reported
    if show_progress and progress_dict is not None and local_progress > 0:
        with progress_dict.get_lock() if hasattr(progress_dict, 'get_lock') else nullcontext():
            progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
    
    return false_positives, times, collided_msgs, messages_processed

class IdMetrics:
    """Class for calculating various metrics for identification systems."""
    
    @staticmethod
    def evaluate_system(
        system: IdSystem,
        num_messages: int = 100000,
        vec_len: int = 16,
        num_validation_messages: int = 1,
        save_interval: int = 1000,
        message_subset_size: int = 10,
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
            message_subset_size: Size of the message subset to consider for compute intensive metrics
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

        fp_rate, false_positives, timing_metrics, collision_metrics, total_messages = IdMetrics._propagate_messages_parallel(
            system, vec_len, num_messages, num_validation_messages, num_processes, show_progress, message_pattern
        )

        # Calculate tag distribution metrics
        sample_messages = list(generate_structured_messages(vec_len, message_pattern, gf_exp, message_subset_size, False))
        tag_metrics = IdMetrics._calculate_tag_metrics(system, sample_messages)
        
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
            
            # Collision metrics
            'num_collisions': collision_metrics['num_collisions'],
            'avg_hamming_distance': collision_metrics['avg_hamming_distance'],
            'min_hamming_distance': collision_metrics['min_hamming_distance'],
            'max_hamming_distance': collision_metrics['max_hamming_distance'],
            'std_hamming_distance': collision_metrics['std_hamming_distance'],
            
            # System characteristics
            'tag_size_bits': float(gf_exp),
            'avg_message_length': message_length,
            'unique_tags': tag_metrics['unique_tags'],
            'tag_uniqueness': tag_metrics['tag_uniqueness'],
            'tag_distribution_uniformity': tag_metrics['tag_distribution_uniformity'],
            'tag_max_value': tag_metrics['tag_max_value'],
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
    ) -> Tuple[float, int, Dict[str, float], Dict[str, float], int]:
        """Memory-optimized parallelized version that generates messages on demand."""
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
        
        # Create a manager to share progress information between processes (only if progress is enabled)
        progress_dict = None
        manager = None
        if show_progress:
            manager = mp.Manager()
            progress_dict = manager.dict()
        
        # Determine update frequency
        update_frequency = max(100, remaining_messages // 100) if show_progress else remaining_messages
        
        # Prepare arguments for worker processes
        worker_args = []
        for i in range(num_processes):
            # Last process gets remainder
            actual_batch_size = batch_size_per_process + (remaining_messages % num_processes if i == num_processes-1 else 0)
            # Each worker gets a different seed to ensure different message sequences
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
                worker_seed  # Add worker seed to ensure different messages
            ))
        
        total_false_positives = 0
        all_times = []
        collided_msgs = []
        total_messages_processed = 1  # Start with 1 for the first message used for codeword
        
        if num_processes == 1 or len(worker_args) == 1:
            # Single process execution for small datasets or when requested
            for args in worker_args:
                fp, times, collided, processed = _worker_generate_and_test(args)
                total_false_positives += fp
                all_times.extend(times)
                collided_msgs.extend(collided)
                total_messages_processed += processed
        else:
            # Multi-process execution
            with mp.Pool(processes=num_processes) as pool:
                if show_progress:
                    # Submit all jobs
                    jobs = [pool.apply_async(_worker_generate_and_test, (args,)) for args in worker_args]
                    
                    desc = f"Processing {num_messages} messages ({num_processes} proc)"
                    sleep_time = 0.1 if num_messages < 10000 else 0.5
                    
                    with tqdm(total=num_messages-1, desc=desc) as pbar:
                        last_total = 0
                        
                        # Monitor progress while jobs are running
                        while any(not job.ready() for job in jobs):
                            current_total = sum(progress_dict.values()) if progress_dict else 0
                            if current_total > last_total:
                                pbar.update(current_total - last_total)
                                last_total = current_total
                            time.sleep(sleep_time)
                        
                        # Get all results
                        for job in jobs:
                            fp, times, collided, processed = job.get()
                            total_false_positives += fp
                            all_times.extend(times)
                            collided_msgs.extend(collided)
                            total_messages_processed += processed
                        
                        # Final update
                        current_total = sum(progress_dict.values()) if progress_dict else 0
                        if current_total > last_total:
                            pbar.update(current_total - last_total)
                else:
                    # No progress bar - just map
                    results = pool.map(_worker_generate_and_test, worker_args)
                    for fp, times, collided, processed in results:
                        total_false_positives += fp
                        all_times.extend(times)
                        collided_msgs.extend(collided)
                        total_messages_processed += processed
            
            # Clean up the manager
            if manager is not None:
                manager.shutdown()
        
        if not all_times:
            return 0.0, total_false_positives, {
                'avg_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'std_execution_time_ms': 0.0
            }, {
                'num_collisions': 0,
                'avg_hamming_distance': 0.0,
                'min_hamming_distance': 0.0,
                'max_hamming_distance': 0.0,
                'std_hamming_distance': 0.0
            }, total_messages_processed
        
        # Calculate false positive rate based on actual messages processed
        fp_rate = total_false_positives / max(1, total_messages_processed - 1)

        # Calculate Hamming distances for collided messages
        hamming_distances = []
        if collided_msgs:
            for msg in collided_msgs:
                hamming_distance = sum(1 for a, b in zip(first_message, msg) if a != b)
                hamming_distances.append(hamming_distance)

        collision_metrics = {
            'num_collisions': len(collided_msgs),
            'avg_hamming_distance': float(np.mean(hamming_distances)) if hamming_distances else 0.0,
            'min_hamming_distance': float(np.min(hamming_distances)) if hamming_distances else 0.0,
            'max_hamming_distance': float(np.max(hamming_distances)) if hamming_distances else 0.0,
            'std_hamming_distance': float(np.std(hamming_distances)) if hamming_distances else 0.0
        }
        
        # Calculate timing metrics
        timing_metrics = {
            'avg_execution_time_ms': float(np.mean(all_times)),
            'min_execution_time_ms': float(np.min(all_times)),
            'max_execution_time_ms': float(np.max(all_times)),
            'std_execution_time_ms': float(np.std(all_times))
        }
        
        return fp_rate, total_false_positives, timing_metrics, collision_metrics, total_messages_processed    

    @staticmethod
    def _propagate_messages(
        system: IdSystem,
        message_set: List[List[int]],
        num_validation_messages: int = 1
    ) -> Tuple[float, int, Dict[str, float]]:
        """Pick first message as message at sender and cycle through the rest of the messages at the receiver while timing the process."""
        n = len(message_set)
        if n < 2:
            raise ValueError("Message set must contain at least two distinct messages for meaningful evaluation.")
        # Use the first message as the one to receive
        first_message = message_set[0]
        false_positives = 0
        times = []

        # Send the first message to get the codeword
        codeword = system.send(first_message)

        for i in range(1, n, num_validation_messages):
            # Time the verification operation
            start_time = time.perf_counter()

            if system.receive_k(codeword, message_set[i:i + num_validation_messages]):
                false_positives += num_validation_messages

            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            times.append(execution_time_ms)
        if not times:
            return 0.0, false_positives, {
                'avg_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'std_execution_time_ms': 0.0
            }
        # Calculate false positive rate
        fp_rate = false_positives / (n - 1) 
        # Calculate timing metrics
        timing_metrics = {
            'avg_execution_time_ms': float(np.mean(times)),
            'min_execution_time_ms': float(np.min(times)),
            'max_execution_time_ms': float(np.max(times)),
            'std_execution_time_ms': float(np.std(times))
        }
        return fp_rate, false_positives, timing_metrics
    
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
            if message == []:
                break  # Stop if empty message is encountered
            tag = system.send(message)
            if isinstance(tag, list):
                # Convert list tags to tuples for consistency
                tags.append(tuple(tag))
            else:
                # For single value tags, just append the value
                tags.append(tag)
        
        if not tags:
            return {
                'unique_tags': 0.0,
                'tag_entropy': 0.0,
                'tag_uniqueness': 0.0,
                'tag_distribution_uniformity': 0.0,
                'tag_max_value': 0.0
            }
        
        # Calculate tag entropy
        tag_counts = Counter(tags)
        total_tags = len(tags)
        tag_entropy = 0.0
        
        # for count in tag_counts.values():
        #     prob = count / total_tags
        #     if prob > 0:
        #         tag_entropy -= prob * math.log2(prob)
        
        # Calculate uniqueness (fraction of unique tags)
        unique_tags = len(tag_counts) if total_tags > 0 else 0
        tag_uniqueness = unique_tags / total_tags if total_tags > 0 else 0.0

        # Calculate distribution uniformity (how close to uniform distribution)
        # This is the relative entropy compared to a uniform distribution
        # D(p_X || p_U) = log2(|χ|) - H(X) where X is the random variable, |χ| the size of the alphabet 
        # uniformity = math.log2(unique_tags) - tag_entropy if unique_tags > 0 else 0.0
        uniformity = 0.0

        tag_max_value = max(tag_counts.keys()) if tag_counts else 0
        
        return {
            'unique_tags': unique_tags,
            'tag_entropy': tag_entropy,
            'tag_uniqueness': tag_uniqueness,
            'tag_distribution_uniformity': uniformity,
            'tag_max_value': tag_max_value
        }
    
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