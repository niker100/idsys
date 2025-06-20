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

from .core import IdSystem, generate_test_messages, create_id_system, _get_idcodes_instance


def _worker_generate_and_test(args):
    """Worker function for generating messages on-demand and testing."""
    system_type, system_params, codeword, vec_len, gf_exp, batch_size, num_validation_messages, progress_dict, update_frequency = args
    
    # Recreate the system in the worker process
    system = create_id_system(system_type, system_params)
    
    # Get idcodes instance for message generation
    idcodes_instance = _get_idcodes_instance(gf_exp)
    
    false_positives = 0
    times = []
    
    # Calculate adjusted vector length
    if gf_exp >= 33:
        vec_len_ = vec_len // 8
    elif gf_exp >= 17:
        vec_len_ = vec_len // 4
    elif gf_exp >= 9:
        vec_len_ = vec_len // 2
    else:
        vec_len_ = vec_len
    
    # For tracking progress with minimal updates
    local_progress = 0
    progress_update_threshold = max(1, min(update_frequency, batch_size // 10))  # Update at most ~10 times per batch
    worker_id = mp.current_process().pid
    
    # Process in mini-batches to save memory while reducing C++ call overhead
    for batch_start in range(0, batch_size, num_validation_messages):
        # Generate validation messages directly
        validation_msgs = []
        for _ in range(min(num_validation_messages, batch_size - batch_start)):
            msg = idcodes_instance.generate_string_sequence(vec_len_)
            validation_msgs.append(msg)
            
        # Time the verification operation
        start_time = time.perf_counter()
        
        if system.receive_k(codeword, validation_msgs):
            false_positives += len(validation_msgs)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        times.append(execution_time_ms)
        
        # Explicitly clear messages to free memory
        validation_msgs.clear()
        
        # Update local progress counter
        messages_processed = min(num_validation_messages, batch_size - batch_start)
        local_progress += messages_processed
        
        # Update shared progress counter infrequently to reduce overhead
        if progress_dict is not None and (local_progress >= progress_update_threshold or batch_start + num_validation_messages >= batch_size):
            progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
            local_progress = 0
    
    # Ensure any remaining progress is reported
    if progress_dict is not None and local_progress > 0:
        progress_dict[worker_id] = progress_dict.get(worker_id, 0) + local_progress
    
    return false_positives, times


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
        num_processes: int = None
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

        Returns:
            Dictionary containing all metrics
        """
        
        # Get system parameters for code rate calculation
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        system_type = type(encoder).__name__.replace('Encoder', '')

        # Calculate approximate message length for code rate
        # Use adjusted length calculation similar to generate_test_messages
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

        fp_rate, false_positives, timing_metrics = IdMetrics._propagate_messages_parallel(
            system, vec_len, num_messages, num_validation_messages, num_processes
        )

        # Calculate entropy metrics
        # entropy_metrics = IdMetrics._calculate_entropy_metrics(message_set[0:message_subset_size], gf_exp)
        
        # Calculate tag distribution metrics
        # tag_metrics = IdMetrics._calculate_tag_metrics(system, message_set[0:message_subset_size])
        
        # Compile comprehensive results
        results = {
            # Core performance metrics
            'false_positive_rate': fp_rate,  # Mean collision probability
            'false_positives': false_positives,
            'code_rate': code_rate,
            
            # Timing metrics
            'avg_execution_time_ms': timing_metrics['avg_execution_time_ms'],
            'min_execution_time_ms': timing_metrics['min_execution_time_ms'],
            'max_execution_time_ms': timing_metrics['max_execution_time_ms'],
            'std_execution_time_ms': timing_metrics['std_execution_time_ms'],
            
            # Efficiency metrics
            'throughput_msgs_per_sec': 1000.0 / timing_metrics['avg_execution_time_ms'] if timing_metrics['avg_execution_time_ms'] > 0 else 0,
            
            # Information theory metrics
            # 'message_entropy': entropy_metrics['message_entropy'],
            # 'tag_entropy': tag_metrics['tag_entropy'],
            # 'compression_ratio': entropy_metrics['message_entropy'] / tag_metrics['tag_entropy'] if tag_metrics['tag_entropy'] > 0 else 0,
            
            # System characteristics
            'tag_size_bits': float(gf_exp),
            'avg_message_length': message_length,
            # 'unique_tags': tag_metrics['unique_tags'],
            # 'tag_uniqueness': tag_metrics['tag_uniqueness'],
            # 'tag_distribution_uniformity': tag_metrics['tag_distribution_uniformity'],
            # 'tag_max_value': tag_metrics['tag_max_value'],        
        }
        
        return results
    
    @staticmethod
    def _calculate_code_rate(system_type: str, avg_message_length: float, gf_exp: int) -> float:
        """Calculate effective code rate defined as the ratio of message bits to tag/output bits."""
        # tag size = gf_exp bits
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
        num_processes: int = None
    ) -> Tuple[float, int, Dict[str, float]]:
        """Memory-optimized parallelized version that generates messages on demand."""
        if num_messages < 2:
            raise ValueError("Need at least two messages for meaningful evaluation.")
        
        # Generate only the first message
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        
        # Get first message for codeword generation
        idcodes = _get_idcodes_instance(gf_exp)
        
        # Calculate adjusted vector length as in generate_test_messages
        if gf_exp >= 33:
            vec_len_ = vec_len // 8
        elif gf_exp >= 17:
            vec_len_ = vec_len // 4
        elif gf_exp >= 9:
            vec_len_ = vec_len // 2
        else:
            vec_len_ = vec_len
        
        # Generate just one message
        first_message = idcodes.generate_string_sequence(vec_len_)
        
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
        
        # Create a manager to share progress information between processes
        manager = mp.Manager()
        progress_dict = manager.dict()
        
        # Determine update frequency - higher for larger batches to minimize overhead
        # Update roughly every ~1% of total messages or at least every 5000 messages
        update_frequency = max(5000, remaining_messages // 100)
        
        # Prepare arguments for worker processes
        worker_args = []
        for i in range(num_processes):
            # Last process gets remainder
            actual_batch_size = batch_size_per_process + (remaining_messages % num_processes if i == num_processes-1 else 0)
            worker_args.append((
                system_type, 
                system_params, 
                codeword, 
                vec_len, 
                gf_exp, 
                actual_batch_size, 
                num_validation_messages,
                progress_dict,  # Pass the shared progress dictionary
                update_frequency  # Update frequency parameter
            ))
        
        total_false_positives = 0
        all_times = []
        
        # Single process is simpler, use direct progress bar
        if num_processes == 1:
            fp, times = 0, []
            with tqdm(total=num_messages-1, desc=f"Processing {num_messages} messages (1 proc)") as pbar:
                for args in worker_args:
                    _fp, _times = _worker_generate_and_test(args)
                    fp += _fp
                    times.extend(_times)
                    pbar.update(args[5])  # Update by batch size
            
            total_false_positives = fp
            all_times = times
            
        else:
            # Multi-process with progress monitoring
            desc = f"Processing {num_messages} messages ({num_processes} proc)"
            
            # Start the pool
            pool = mp.Pool(processes=num_processes)
            
            # Submit all jobs
            jobs = [pool.apply_async(_worker_generate_and_test, (args,)) for args in worker_args]
            
            # Monitor progress while jobs are running
            # Calculate sleep time based on expected total runtime
            # For short runs (<1M messages), check more frequently
            sleep_time = 0.5 if num_messages < 1000000 else 1.0
            
            with tqdm(total=num_messages-1, desc=desc) as pbar:
                # Track the last reported count
                last_total = 0
                
                # Continue until all jobs complete
                while any(not job.ready() for job in jobs):
                    # Calculate current progress
                    current_total = sum(progress_dict.values())
                    
                    # Update progress bar with the difference
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total
                    
                    time.sleep(sleep_time)  # Reduced check frequency
                
                # Get all results
                for job in jobs:
                    fp, times = job.get()
                    total_false_positives += fp
                    all_times.extend(times)
                
                # Final update to ensure bar reaches 100%
                current_total = sum(progress_dict.values())
                if current_total > last_total:
                    pbar.update(current_total - last_total)
            
            # Close and join the pool
            pool.close()
            pool.join()
            
            # Clean up the manager
            manager.shutdown()
        
        if not all_times:
            return 0.0, total_false_positives, {
                'avg_execution_time_ms': 0.0,
                'min_execution_time_ms': 0.0,
                'max_execution_time_ms': 0.0,
                'std_execution_time_ms': 0.0
            }
        
        # Calculate false positive rate
        fp_rate = total_false_positives / (num_messages - 1) 
        
        # Calculate timing metrics
        timing_metrics = {
            'avg_execution_time_ms': float(np.mean(all_times)),
            'min_execution_time_ms': float(np.min(all_times)),
            'max_execution_time_ms': float(np.max(all_times)),
            'std_execution_time_ms': float(np.std(all_times))
        }
        
        return fp_rate, total_false_positives, timing_metrics
    
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
            tag = system.send(message)
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
        
        for count in tag_counts.values():
            prob = count / total_tags
            if prob > 0:
                tag_entropy -= prob * math.log2(prob)
        
        # Calculate uniqueness (fraction of unique tags)
        unique_tags = len(tag_counts) if total_tags > 0 else 0
        tag_uniqueness = unique_tags / total_tags if total_tags > 0 else 0.0

        # Calculate distribution uniformity (how close to uniform distribution)
        # This is the relative entropy compared to a uniform distribution
        # D(p_X || p_U) = log2(|χ|) - H(X) where X is the random variable, |χ| the size of the alphabet 
        uniformity = math.log2(unique_tags) - tag_entropy if unique_tags > 0 else 0.0

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
        num_processes: int = None
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
                num_processes=num_processes
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