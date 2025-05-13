#!/usr/bin/env python3
"""
Metrics module for evaluating identification systems.

This module provides functions and classes for measuring the performance
of identification systems based on various metrics, such as reliability,
efficiency, error rates, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time
import math
from collections import defaultdict

from .core import IdSystem, IdEncoder, IdDecoder


class IdMetrics:
    """Class for calculating various metrics for identification systems."""
    
    @staticmethod
    def reliability(system: IdSystem, message_set: List[Any], num_trials: int = 1000) -> float:
        """
        Calculate the reliability of the identification system.
        
        Reliability is the probability of correct identification when the
        receiver message matches the sender message, and correct rejection
        when it doesn't.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            num_trials: Number of trials for the Monte Carlo simulation
            
        Returns:
            float: The reliability score (0-1, higher is better)
        """
        correct_count = 0
        
        for _ in range(num_trials):
            # Choose a random message from the set as the sender's message
            sender_idx = np.random.randint(0, len(message_set))
            sender_message = message_set[sender_idx]
            
            # Choose whether this is a true or false identification scenario
            is_true_id = np.random.choice([True, False])
            
            if is_true_id:
                # Receiver has the same message
                receiver_message = sender_message
            else:
                # Receiver has a different message
                receiver_idx = np.random.randint(0, len(message_set))
                while receiver_idx == sender_idx:
                    receiver_idx = np.random.randint(0, len(message_set))
                receiver_message = message_set[receiver_idx]
            
            # Encode and send message
            codeword = system.send(sender_message)
            
            # Decide at the receiver
            is_identified = system.receive(codeword, receiver_message)
            
            # Check if decision is correct
            if (is_true_id and is_identified) or (not is_true_id and not is_identified):
                correct_count += 1
                
        return correct_count / num_trials
    
    @staticmethod
    def error_rates(system: IdSystem, message_set: List[Any], num_trials: int = 1000) -> Dict[str, float]:
        """
        Calculate error rates for the identification system.
        
        This calculates:
        - False positive rate: Probability of identifying a non-matching message as matching
        - False negative rate: Probability of failing to identify a matching message
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            num_trials: Number of trials for the Monte Carlo simulation
            
        Returns:
            Dict with error rates
        """
        false_positives = 0
        false_negatives = 0
        true_id_trials = 0
        false_id_trials = 0
        
        for _ in range(num_trials):
            # Choose a random message from the set as the sender's message
            sender_idx = np.random.randint(0, len(message_set))
            sender_message = message_set[sender_idx]
            
            # Choose whether this is a true or false identification scenario
            is_true_id = np.random.choice([True, False])
            
            if is_true_id:
                # Receiver has the same message
                receiver_message = sender_message
                true_id_trials += 1
            else:
                # Receiver has a different message
                receiver_idx = np.random.randint(0, len(message_set))
                while receiver_idx == sender_idx:
                    receiver_idx = np.random.randint(0, len(message_set))
                receiver_message = message_set[receiver_idx]
                false_id_trials += 1
            
            # Encode and send message
            codeword = system.send(sender_message)
            
            # Decide at the receiver
            is_identified = system.receive(codeword, receiver_message)
            
            # Check for errors
            if is_true_id and not is_identified:
                false_negatives += 1
            elif not is_true_id and is_identified:
                false_positives += 1
        
        # Calculate rates
        false_positive_rate = false_positives / false_id_trials if false_id_trials > 0 else 0
        false_negative_rate = false_negatives / true_id_trials if true_id_trials > 0 else 0
        
        return {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate
        }
    
    @staticmethod
    def worst_case_collision_probability(system: IdSystem, message_set: List[Any], 
                                         sample_size: Optional[int] = None, 
                                         num_trials: int = 100) -> float:
        """
        Estimate the worst-case probability of collision.
        
        For any message in the set, find the highest probability of another
        message being incorrectly identified as matching.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            sample_size: Number of messages to sample (None = use all messages)
            num_trials: Number of trials for each message pair
            
        Returns:
            float: The worst-case collision probability (0-1, lower is better)
        """
        if sample_size is None or sample_size > len(message_set):
            sample_size = len(message_set)
        
        # Sample messages to test
        sampled_indices = np.random.choice(len(message_set), size=sample_size, replace=False)
        sampled_messages = [message_set[i] for i in sampled_indices]
        
        worst_collision_prob = 0
        
        for sender_message in sampled_messages:
            # Encode sender message
            codeword = system.send(sender_message)
            
            # Compare with all other messages
            collisions = 0
            for receiver_message in message_set:
                if receiver_message == sender_message:
                    continue
                
                # Check if the receiver would identify this message as matching
                if system.receive(codeword, receiver_message):
                    collisions += 1
            
            collision_prob = collisions / (len(message_set) - 1) if len(message_set) > 1 else 0
            worst_collision_prob = max(worst_collision_prob, collision_prob)
        
        return worst_collision_prob
    
    @staticmethod
    def efficiency(system: IdSystem) -> Dict[str, float]:
        """
        Calculate efficiency metrics for the identification system.
        
        This measures:
        - Code rate: The ratio of message size to codeword size
        - Encoding time: The time taken to encode a message
        
        Args:
            system: The identification system to evaluate
            
        Returns:
            Dict with efficiency metrics
        """
        # For measuring code rate, we need to know message and codeword sizes
        # Let's use a simple numeric message as a test case
        test_message = 12345
        
        # Encode and measure codeword size
        start_time = time.time()
        codeword = system.send(test_message)
        end_time = time.time()
        
        # Message size in bits (estimated)
        message_size_bits = math.ceil(math.log2(test_message + 1)) if test_message > 0 else 1
        
        # Codeword size in bits
        if isinstance(codeword, np.ndarray):
            codeword_bits = codeword.size * math.ceil(math.log2(np.max(codeword) + 1)) if codeword.size > 0 else 1
        else:
            # If not a numpy array, estimate size by serializing
            codeword_bits = len(str(codeword)) * 8  # Rough estimate
        
        # Calculate code rate (message bits / codeword bits)
        code_rate = message_size_bits / codeword_bits if codeword_bits > 0 else 0
        
        # Encoding time
        encoding_time = end_time - start_time
        
        return {
            "code_rate": code_rate,
            "encoding_time_ms": encoding_time * 1000
        }
    
    @staticmethod
    def collision_matrix(system: IdSystem, message_set: List[Any], max_messages: int = 20) -> np.ndarray:
        """
        Create a collision matrix showing which messages collide.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            max_messages: Maximum number of messages to include in the matrix
            
        Returns:
            np.ndarray: A boolean matrix where entry (i, j) is True if message i and j collide
        """
        n_messages = min(len(message_set), max_messages)
        collision_mat = np.zeros((n_messages, n_messages), dtype=bool)
        
        # Calculate codewords for all messages
        codewords = [system.send(message_set[i]) for i in range(n_messages)]
        
        # Check for collisions
        for i in range(n_messages):
            for j in range(i + 1, n_messages):
                # Check if these messages would be confused with each other
                collide_ij = system.receive(codewords[i], message_set[j])
                collide_ji = system.receive(codewords[j], message_set[i])
                
                collision_mat[i, j] = collide_ij
                collision_mat[j, i] = collide_ji
                
        return collision_mat
    
    @staticmethod
    def soft_verification_rate(system: IdSystem, message_set: List[Any], 
                               distance_func: Callable[[Any, Any], float],
                               threshold: float = 0.8,
                               num_trials: int = 100) -> float:
        """
        Calculate the soft verification rate using a distance function.
        
        Instead of binary decision, this uses a distance function to determine
        if two messages are "close enough" to be considered related.
        
        Args:
            system: The identification system to evaluate
            message_set: The set of possible messages
            distance_func: Function to calculate distance between messages (0-1, lower is closer)
            threshold: Distance threshold for considering messages "close"
            num_trials: Number of trials
            
        Returns:
            float: Rate at which close messages are correctly identified as related
        """
        close_pairs_count = 0
        identified_close_pairs = 0
        
        for _ in range(num_trials):
            # Choose random sender message
            sender_idx = np.random.randint(0, len(message_set))
            sender_message = message_set[sender_idx]
            
            # Choose random receiver message
            receiver_idx = np.random.randint(0, len(message_set))
            receiver_message = message_set[receiver_idx]
            
            # Calculate distance between messages
            distance = distance_func(sender_message, receiver_message)
            is_close = distance < threshold
            
            if is_close:
                close_pairs_count += 1
                
                # Encode and send message
                codeword = system.send(sender_message)
                
                # Check if receiver would identify the message
                is_identified = system.receive(codeword, receiver_message)
                if is_identified:
                    identified_close_pairs += 1
        
        # Calculate soft verification rate
        rate = identified_close_pairs / close_pairs_count if close_pairs_count > 0 else 0
        
        return rate