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
from collections import Counter, defaultdict

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
                                        num_trials: int = 1000) -> float:
        """
        Estimate the worst-case probability of tag collision (i.e., two different messages
        produce the same tag at the same random position pi).

        For a strong RS tagging code and a large message space, this should be extremely low;
        if the tag length is L bytes, the theoretical random collision probability is about 1/256^L.

        Returns the maximum observed collision rate over all sampled messages as an upper bound.

        NOTE: For well-designed RS tag codes, this will often be zero in practice.
        """
        if sample_size is None or sample_size > len(message_set):
            sample_size = min(len(message_set), 100)

        sampled_indices = np.random.choice(len(message_set), size=sample_size, replace=False)
        sampled_messages = [message_set[i] for i in sampled_indices]
        collision_probabilities = []

        for sender_message in sampled_messages:
            collisions_per_trial = []
            for _ in range(num_trials):
                codeword = system.send(sender_message)
                pi, encoded_tag = codeword

                # Sample a subset of receiver messages for larger spaces
                max_test = min(100, len(message_set)-1)
                test_indices = np.random.choice([i for i in range(len(message_set)) if message_set[i] != sender_message], size=max_test, replace=False)
                test_messages = [message_set[i] for i in test_indices]

                collisions = 0
                for receiver_message in test_messages:
                    try:
                        recomputed_tag = system.decoder._recompute_tag(receiver_message, pi)
                        if np.array_equal(encoded_tag, recomputed_tag):
                            collisions += 1
                    except Exception:
                        continue
                collision_rate = collisions / len(test_messages) if test_messages else 0
                collisions_per_trial.append(collision_rate)
            # Take 95th percentile for conservative bound
            collision_probabilities.append(np.percentile(collisions_per_trial, 95))

        return max(collision_probabilities) if collision_probabilities else 0
    
    @staticmethod
    def efficiency(system: IdSystem) -> Dict[str, float]:
        """
        Calculate efficiency metrics for the identification system.
        
        Args:
            system: The identification system to evaluate
            
        Returns:
            Dict with efficiency metrics
        """
        # For RS-based systems, get exact code rate from parameters
        if hasattr(system.encoder, 'parameters') and 'message_length' in system.encoder.parameters and 'nsym' in system.encoder.parameters:
            message_length = system.encoder.parameters["message_length"]
            nsym = system.encoder.parameters["nsym"]            
            code_length = system.encoder.parameters["code_length"]
            # Exact code rate calculation for Reed-Solomon
            code_rate = (message_length + nsym) / code_length
            effective_code_rate = (message_length) / code_length           
        else:
            # Fallback to approximation if system doesn't expose RS parameters
            test_message = "12345"
            start_time = time.time()
            codeword = system.send(test_message)
            end_time = time.time()
            
            # Message size in bits (estimated)
            message_size_bits = math.ceil(math.log2(test_message + 1)) if test_message > 0 else 1
            
            # Codeword size in bits
            if isinstance(codeword, tuple) and len(codeword) == 2:
                _, tag = codeword
                if isinstance(tag, np.ndarray):
                    codeword_bits = tag.size * 8  # Assuming 8 bits per byte
                else:
                    codeword_bits = len(str(tag)) * 8  # Rough estimate
            elif isinstance(codeword, np.ndarray):
                codeword_bits = codeword.size * 8
            else:
                codeword_bits = len(str(codeword)) * 8
            
            code_rate = message_size_bits / codeword_bits if codeword_bits > 0 else 0
            effective_code_rate = code_rate
        
        # Encoding time measurement
        start_time = time.time()
        system.send("12345")  # Use a consistent test message
        encoding_time = time.time() - start_time
        
        return {
            "code_rate": code_rate,
            "effective_code_rate": effective_code_rate,
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


class MessageAnalysisMetrics:
    """Class for analyzing message characteristics like entropy."""
    
    @staticmethod
    def calculate_message_entropy(messages: List[str]) -> Tuple[float, float]:
        """
        Calculate the entropy of a set of messages.
        
        Entropy measures the information content in the messages. Higher entropy
        indicates more information density and less predictability.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Tuple of (character entropy, message entropy)
        """
        all_chars = ''.join(messages)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        # Calculate entropy using Shannon's formula
        char_entropy = -sum((count / total_chars) * math.log2(count / total_chars) 
                          for count in char_counts.values())
        
        # Message entropy is character entropy times the average message length
        avg_length = sum(len(m) for m in messages) / len(messages)
        msg_entropy = char_entropy * avg_length
        
        return char_entropy, msg_entropy
    
    @staticmethod
    def analyze_alphabet_usage(messages: List[str]) -> Dict[str, float]:
        """
        Analyze how the alphabet is used in the messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with alphabet usage statistics
        """
        all_chars = ''.join(messages)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        # Calculate character frequencies
        frequencies = {char: count/total_chars for char, count in char_counts.items()}
        
        # Calculate additional statistics
        unique_chars = len(char_counts)
        avg_freq = 1.0 / unique_chars
        freq_variance = sum((freq - avg_freq)**2 for freq in frequencies.values()) / unique_chars
        
        return {
            'frequencies': frequencies,
            'unique_chars': unique_chars,
            'frequency_variance': freq_variance,
            'most_common': char_counts.most_common(5)
        }

class TaggingMetrics:
    """Metrics specific to tagging systems."""
    
    @staticmethod
    def tag_rate(system: IdSystem) -> float:
        """
        Calculate the tag rate (tag length / message length).
        
        Args:
            system: The tagging system to evaluate
            
        Returns:
            Tag rate as a float
        """
        if not hasattr(system.encoder, 'parameters'):
            return 0.0
            
        params = system.encoder.parameters
        message_length = params.get('message_length', 0)
        code_length = params.get('code_length', 0)
        
        return code_length / message_length if message_length > 0 else 0.0
    
    @staticmethod
    def effective_tag_rate(system: IdSystem) -> float:
        """
        Calculate the effective tag rate considering Reed-Solomon overhead.
        
        Args:
            system: The tagging system to evaluate
            
        Returns:
            Effective tag rate as a float
        """
        if not hasattr(system.encoder, 'parameters'):
            return 0.0
            
        params = system.encoder.parameters
        message_length = params.get('message_length', 0)
        code_length = params.get('code_length', 0)
        nsym = params.get('nsym', 0)
        
        return (code_length - nsym) / message_length if message_length > 0 else 0.0
    
    @staticmethod
    def analyze_tag_distribution(system: IdSystem, messages: List[str], num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze the distribution of tags produced by the system.
        
        Args:
            system: The tagging system to evaluate
            messages: List of messages to analyze
            num_samples: Number of tag samples to generate
            
        Returns:
            Dictionary with tag distribution statistics
        """
        tags = []
        positions = []
        
        # Sample tags from random messages
        for _ in range(num_samples):
            msg = np.random.choice(messages)
            try:
                pos, tag = system.send(msg)
                tags.append(tuple(tag))
                positions.append(pos)
            except Exception:
                continue
                
        # Analyze tag statistics
        unique_tags = len(set(tags))
        unique_positions = len(set(positions))
        
        return {
            'unique_tags': unique_tags,
            'unique_positions': unique_positions,
            'tag_entropy': -sum((tags.count(t)/len(tags) * math.log2(tags.count(t)/len(tags))) 
                               for t in set(tags)) if tags else 0,
            'position_entropy': -sum((positions.count(p)/len(positions) * math.log2(positions.count(p)/len(positions))) 
                                   for p in set(positions)) if positions else 0
        }