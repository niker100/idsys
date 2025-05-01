import numpy as np
import time
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple


class Metrics:
    """
    Class containing static methods for calculating essential metrics
    to evaluate identification system performance.
    Focuses on mean error rate, maximum error, reliability, efficiency, and security.
    """
    
    @staticmethod
    def reliability(successful_identifications, total_attempts):
        """
        Calculate the reliability of an identification system.
        
        Args:
            successful_identifications (int): Number of correct identifications
            total_attempts (int): Total number of identification attempts
            
        Returns:
            float: Reliability as a fraction between 0 and 1
        """
        if total_attempts == 0:
            return 0
        return successful_identifications / total_attempts
    
    @staticmethod
    def error_rate(errors, total_attempts):
        """
        Calculate the error rate of an identification system.
        
        Args:
            errors (int): Number of incorrect identifications
            total_attempts (int): Total number of identification attempts
            
        Returns:
            float: Error rate as a fraction between 0 and 1
        """
        if total_attempts == 0:
            return 0
        return errors / total_attempts
    
    @staticmethod
    def theoretical_collision_probability(tag_length: int, num_messages: int) -> float:
        """
        Calculate the theoretical probability of at least one collision
        when mapping n messages to 2^tag_length possible tags.
        Uses the birthday problem formula.
        
        Args:
            tag_length: Length of the tag in bits
            num_messages: Number of messages
            
        Returns:
            float: Probability of at least one collision
        """
        m = 2 ** tag_length  # Number of possible tags
        n = num_messages     # Number of messages
        
        # Handle edge cases
        if n <= 1:
            return 0.0
        if n > m:
            return 1.0
            
        # Use the birthday problem formula:
        # For large m, approximated as: P(collision) ≈ 1 - exp(-n(n-1)/2m)
        return 1 - np.exp(-n * (n-1) / (2 * m))
    
    @staticmethod
    def required_tag_length(num_messages: int, target_collision_prob: float = 1e-6) -> int:
        """
        Calculate the minimum tag length required to achieve a target collision probability.
        
        Args:
            num_messages: Number of messages to encode
            target_collision_prob: Target collision probability (default: 1e-6)
            
        Returns:
            int: Required tag length in bits
        """
        # We need to find m such that 1 - exp(-n(n-1)/2m) <= target_p
        n = num_messages
        if n <= 1:
            return 1  # Minimum length
            
        # Calculate required number of possible tags
        try:
            m = -n * (n-1) / (2 * np.log(1 - target_collision_prob))
        except (OverflowError, ValueError):
            # Fallback if we encounter numerical issues
            m = n**2 / (2 * target_collision_prob)
        
        # Calculate required tag length (log2(m) rounded up)
        return max(1, int(np.ceil(np.log2(m))))
    
    @staticmethod
    def efficiency(message_size: int, tag_length: int) -> float:
        """
        Calculate the identification efficiency (compression factor).
        
        Args:
            message_size (int): Size of the original message in bits
            tag_length (int): Length of the tag in bits
            
        Returns:
            float: Efficiency factor (higher is better)
        """
        return message_size / tag_length
    
    @staticmethod
    def security_level(tag_length: int) -> Dict[str, Any]:
        """
        Assess the security level of a tag length.
        
        Args:
            tag_length (int): Length of the tag in bits
            
        Returns:
            dict: Security assessment
        """
        # Security comparison to common standards
        security_level = {
            'collision_resistance': min(tag_length / 2, 128), # Ideal collision resistance is half the bit length
            'max_secure_messages': 2**(tag_length/2),  # Square root bound for collision resistance
            'comparable_to': Metrics._get_security_comparison(tag_length)
        }
        
        return security_level
    
    @staticmethod
    def _get_security_comparison(tag_length: int) -> str:
        """Helper method to get security comparison"""
        if tag_length < 32:
            return "Weak - not suitable for security applications"
        elif tag_length < 64:
            return "Moderate - suitable for non-critical applications"
        elif tag_length < 128:
            return "Strong - suitable for most applications"
        else:
            return "Very strong - suitable for high-security applications"


class Evaluator:
    """
    Class for evaluating an identification system using essential metrics.
    Focuses on mean error rate, maximum error, reliability, efficiency, and security.
    """
    
    def __init__(self, identification_system):
        """
        Initialize the evaluator with an identification system.
        
        Args:
            identification_system: The identification system to evaluate
        """
        self.system = identification_system
        self.results = defaultdict(list)
        
    def evaluate_pair(self, sender_message, receiver_message, expected_match=None):
        """
        Evaluate a single identification attempt.
        
        Args:
            sender_message: The message from the sender (Alice)
            receiver_message: The message the receiver (Bob) wants to check
            expected_match (bool, optional): The expected match result
            
        Returns:
            dict: Results of the evaluation
        """
        start_time = time.time()
        encoded_message, is_match = self.system.run_identification(sender_message, receiver_message)
        end_time = time.time()
        
        # Calculate message sizes
        original_size = len(str(sender_message).encode('utf-8'))
        
        # Handle different types of encoded messages
        if isinstance(encoded_message, int):
            tag_length = getattr(self.system.encoder, 'tag_length', encoded_message.bit_length())
        else:
            tag_length = getattr(self.system.encoder, 'tag_length', None)
        
        result = {
            'sender_message': sender_message,
            'receiver_message': receiver_message,
            'is_match': is_match,
            'processing_time': end_time - start_time,
            'original_size': original_size,
        }
        
        # Add efficiency calculation if tag length is available
        if tag_length is not None:
            result['efficiency'] = Metrics.efficiency(original_size * 8, tag_length)
        
        # Add expected match results if provided
        if expected_match is not None:
            result['expected_match'] = expected_match
            result['correct_identification'] = is_match == expected_match
        
        self.results['evaluations'].append(result)
        return result
    
    def evaluate_dataset(self, message_pairs, expected_matches=None):
        """
        Evaluate multiple identification attempts using a dataset.
        
        Args:
            message_pairs (list): List of (sender_message, receiver_message) tuples
            expected_matches (list, optional): List of expected match results
            
        Returns:
            dict: Aggregate results and metrics
        """
        self.results = defaultdict(list)
        
        for i, (sender_message, receiver_message) in enumerate(message_pairs):
            expected_match = None if expected_matches is None else expected_matches[i]
            self.evaluate_pair(sender_message, receiver_message, expected_match)
            
        return self.get_aggregate_results()
    
    def get_aggregate_results(self):
        """
        Calculate aggregate metrics from all evaluations.
        
        Returns:
            dict: Aggregate metrics focused on key performance indicators
        """
        if not self.results.get('evaluations', []):
            return {'error': 'No evaluations have been performed'}
            
        evals = self.results['evaluations']
        
        # Count matches and errors
        total = len(evals)
        
        # Handle error rates and reliability
        if all('expected_match' in r for r in evals):
            # When we have expected match data:
            correct = sum(1 for r in evals if r['correct_identification'])
            errors = total - correct
            mean_error_rate = Metrics.error_rate(errors, total)
            reliability = Metrics.reliability(correct, total)
            
            # Calculate false positives and negatives
            true_negatives = sum(1 for r in evals if r.get('expected_match') is False and r['is_match'] is False)
            false_positives = sum(1 for r in evals if r.get('expected_match') is False and r['is_match'] is True)
            true_positives = sum(1 for r in evals if r.get('expected_match') is True and r['is_match'] is True)
            false_negatives = sum(1 for r in evals if r.get('expected_match') is True and r['is_match'] is False)
            
            # Maximum error among false positive and false negative rates
            total_negatives = true_negatives + false_positives
            total_positives = true_positives + false_negatives
            
            fp_rate = Metrics.error_rate(false_positives, total_negatives) if total_negatives > 0 else 0
            fn_rate = Metrics.error_rate(false_negatives, total_positives) if total_positives > 0 else 0
            max_error_rate = max(fp_rate, fn_rate)
            
            # Create confusion matrix
            confusion_matrix = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives
            }
        else:
            # Without expected match data, we can't calculate error rates
            reliability = None
            mean_error_rate = None
            max_error_rate = None
            confusion_matrix = None
        
        # Get tag length if available
        tag_length = getattr(self.system.encoder, 'tag_length', None)
        
        # Calculate security metrics if we have tag length
        security_metrics = None
        if tag_length is not None:
            distinct_messages = len({str(r['sender_message']) for r in evals})
            collision_prob = Metrics.theoretical_collision_probability(tag_length, distinct_messages)
            security_metrics = Metrics.security_level(tag_length)
            security_metrics['collision_probability'] = collision_prob
        
        # Calculate average efficiency
        avg_efficiency = None
        if all('efficiency' in r for r in evals):
            avg_efficiency = np.mean([r['efficiency'] for r in evals])
        
        # Prepare the results dictionary
        metrics = {
            'num_evaluations': total,
            'avg_processing_time': np.mean([r['processing_time'] for r in evals]),
            'encoder_type': str(type(self.system.encoder).__name__)
        }
        
        # Add tag length and security metrics if available
        if tag_length is not None:
            metrics['tag_length'] = tag_length
            metrics['security'] = security_metrics
        
        # Add efficiency metrics if available
        if avg_efficiency is not None:
            metrics['avg_efficiency'] = avg_efficiency
        
        # Add error and reliability metrics if available
        if reliability is not None:
            metrics.update({
                'reliability': reliability,
                'mean_error_rate': mean_error_rate,
                'max_error_rate': max_error_rate,
                'confusion_matrix': confusion_matrix
            })
            
        return metrics
    
    def analyze_parameter_impact(self, tag_lengths, message_sizes):
        """
        Analyze how different parameters impact identification performance.
        
        Args:
            tag_lengths (list): List of tag lengths to test
            message_sizes (list): List of message sizes to test
            
        Returns:
            dict: Analysis of parameter impacts on key metrics
        """
        results = {
            'tag_lengths': tag_lengths,
            'message_sizes': message_sizes,
            'efficiency': {},
            'security': {},
            'reliability_estimate': {}
        }
        
        for tag_length in tag_lengths:
            results['efficiency'][tag_length] = {}
            results['security'][tag_length] = Metrics.security_level(tag_length)
            
            for msg_size in message_sizes:
                # Calculate efficiency
                results['efficiency'][tag_length][msg_size] = Metrics.efficiency(msg_size * 8, tag_length)
                
                # Estimate reliability based on collision probability
                # Use a smaller number for distinct_messages to keep the calculation realistic
                distinct_messages = min(2**16, msg_size * 8)  # Avoid overflow
                collision_prob = Metrics.theoretical_collision_probability(tag_length, distinct_messages)
                
                if tag_length not in results['reliability_estimate']:
                    results['reliability_estimate'][tag_length] = {}
                
                results['reliability_estimate'][tag_length][msg_size] = 1 - collision_prob
        
        return results