"""
Core metrics functionality for identification system evaluation.

This module provides the main IdMetrics class for evaluating 
identification system performance.

Copyright (c) 2025 Nick Schubert
Licensed under the MIT License - see LICENSE file for details
"""

from typing import Dict, Any
from .idsystems import IdSystem
from .statistics import StatisticsUtils
from .parallel import propagate_messages_parallel


class IdMetrics:
    """
    Class for calculating various metrics for identification systems.
    
    This class provides methods for comprehensive evaluation of identification
    systems, calculating metrics such as false positive rates, timing statistics,
    and code rates.
    
    Example:
        >>> from idsys.core import create_id_system
        >>> system = create_id_system("RSID", {"gf_exp": 8})
        >>> metrics = IdMetrics.evaluate_system(system, num_messages=10000)
        >>> print(f"False positive rate: {metrics['false_positive_rate']}")
        >>> print(f"Average execution time: {metrics['avg_execution_time_ms']} ms")
    """
        
    @staticmethod
    def evaluate_system(
        system: IdSystem,
        num_messages: int = 100000,
        vec_len: int = 16,
        num_validation_messages: int = 1,
        num_processes: int = None,
        message_pattern: str = 'random',
        calculate_pdfs: bool = False
    ) -> Dict[str, Any]:
        """
        Complete evaluation of an identification system.
        
        This method calculates comprehensive metrics for an identification system,
        including performance, efficiency, and statistical characteristics.
        
        Args:
            system: The identification system to evaluate
            num_messages: Number of messages to generate for evaluation
            vec_len: Length of the messages in bytes
            num_validation_messages: Number of valid messages at the receiver for k-identification
            num_processes: Number of processes to use for parallelization (None for auto)
            message_pattern: Pattern for message generation ('random', 'incremental', etc.)
            calculate_pdfs: Whether to calculate message and tag PDFs

        Returns:
            Dictionary containing all metrics:
            - false_positive_rate: Rate of false positive identifications
            - total_messages: Total number of messages processed
            - false_positives: Total number of false positives
            - code_rate_bulk: Code rate for bulk transmission
            - code_rate_subsequently: Code rate for sequential transmission
            - theoretical_fp_rate: Theoretical false positive rate
            - avg_execution_time_ms: Average execution time in milliseconds
            - min_execution_time_ms: Minimum execution time observed
            - max_execution_time_ms: Maximum execution time observed
            - std_execution_time_ms: Standard deviation of execution times
            - throughput_msgs_per_sec: Messages processed per second
            - message_pdf: Probability density function of message symbols
            - avg_hamming_distance: Average Hamming distance between messages
            - min_hamming_distance: Minimum Hamming distance observed
            - max_hamming_distance: Maximum Hamming distance observed
            - std_hamming_distance: Standard deviation of Hamming distances
            - collisions_*: Metrics for collided messages
            - tag_size_bits: Size of tags in bits
            - tag_pdf: Probability density function of tag symbols
            
        Example:
            >>> system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
            >>> metrics = IdMetrics.evaluate_system(system, num_messages=1000)
            >>> print(f"False positive rate: {metrics['false_positive_rate']}")
        """
        
        # Get system parameters for code rate calculation
        encoder = system.encoder
        params = getattr(encoder, 'parameters', {})
        gf_exp = params.get('gf_exp', 8)
        num_tags = len(params.get('tag_pos', [0])) if isinstance(params.get('tag_pos'), list) else 1

        # Calculate approximate message length in symbols based on GF exponent
        if gf_exp >= 33:
            message_length = vec_len // 8
        elif gf_exp >= 17:
            message_length = vec_len // 4
        elif gf_exp >= 9:
            message_length = vec_len // 2
        else:
            message_length = vec_len
        
        # Calculate code rates
        code_rate_bulk, coderate_subsequently = StatisticsUtils.calculate_code_rate(
            num_tags, vec_len*8, gf_exp
        )

        # Calculate theoretical false positive rate
        theoretical_fp_rate = StatisticsUtils.calculate_theoretical_fp_rate(
            num_validation_messages, num_tags, gf_exp
        )

        # Run parallel message processing with integrated metrics calculation
        fp_rate, false_positives, timing_metrics, collision_metrics, total_messages, aggregated_metrics = (
            propagate_messages_parallel(
                system, vec_len, num_messages, num_validation_messages, 
                num_processes, message_pattern, calculate_pdfs
            )
        )

        # Compile comprehensive results
        results = {
            # Core performance metrics
            'total_messages': total_messages,
            'false_positive_rate': fp_rate,
            'false_positives': false_positives,
            'code_rate_bulk': code_rate_bulk,
            'code_rate_subsequently': coderate_subsequently,
            'theoretical_fp_rate': theoretical_fp_rate,
            
            # Timing metrics
            'avg_execution_time_ms': timing_metrics['avg_execution_time_ms'],
            'min_execution_time_ms': timing_metrics['min_execution_time_ms'],
            'max_execution_time_ms': timing_metrics['max_execution_time_ms'],
            'std_execution_time_ms': timing_metrics['std_execution_time_ms'],
            
            # Efficiency metrics
            'throughput_msgs_per_sec': 1000.0 / timing_metrics['avg_execution_time_ms'] 
                                      if timing_metrics['avg_execution_time_ms'] > 0 else 0,
            
            # Message set characteristics            
            'message_pdf': aggregated_metrics['message_pdf'] if calculate_pdfs else {},
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
            'tag_pdf': aggregated_metrics['tag_pdf'] if calculate_pdfs else {},
        }
        
        return results