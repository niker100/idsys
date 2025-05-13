#!/usr/bin/env python3
"""
Test suite for identification system framework.

This module contains unit tests and integration tests for the
identification system components and metrics.
"""

import unittest
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Any

from .core import (
    IdEncoder, IdDecoder, IdSystem, 
    HashTaggingEncoder, BitwiseCompareDecoder,
    RandomProjectionEncoder, HammingDistanceDecoder,
    create_id_system, generate_numeric_messages, generate_string_messages
)
from .metrics import IdMetrics
from .visualization import IdVisualizer


class TestEncoders(unittest.TestCase):
    """Test encoder implementations."""
    
    def test_hash_tagging_encoder(self):
        """Test that the hash tagging encoder produces valid codewords."""
        # Create encoder with different code lengths
        for code_length in [4, 8, 16, 32]:
            encoder = HashTaggingEncoder({"code_length": code_length})
            
            # Generate some test messages
            messages = ["test1", "test2", 12345, "hello world"]
            
            for message in messages:
                # Encode the message
                codeword = encoder.encode(message)
                
                # Check that the codeword has the right length
                self.assertEqual(len(codeword), code_length)
                
                # Check that the codeword contains only binary values
                self.assertTrue(np.all((codeword == 0) | (codeword == 1)))
                
                # Check that encoding is deterministic
                codeword2 = encoder.encode(message)
                np.testing.assert_array_equal(codeword, codeword2)
                
                # Check that different messages produce different codewords
                for other_message in messages:
                    if other_message != message:
                        other_codeword = encoder.encode(other_message)
                        self.assertFalse(np.array_equal(codeword, other_codeword))
    
    def test_random_projection_encoder(self):
        """Test that the random projection encoder works correctly."""
        # Create encoder
        encoder = RandomProjectionEncoder({
            "code_length": 16,
            "feature_size": 32,
            "seed": 42  # Use fixed seed for reproducibility
        })
        
        # Generate some test messages
        messages = ["test1", "test2", 12345, "hello world"]
        
        for message in messages:
            # Encode the message
            codeword = encoder.encode(message)
            
            # Check that the codeword has the right length
            self.assertEqual(len(codeword), 16)
            
            # Check that the codeword contains only binary values
            self.assertTrue(np.all((codeword == 0) | (codeword == 1)))
            
            # Check that encoding is deterministic
            codeword2 = encoder.encode(message)
            np.testing.assert_array_equal(codeword, codeword2)
            
            # Check that changing the seed changes the codeword
            encoder_diff_seed = RandomProjectionEncoder({
                "code_length": 16,
                "feature_size": 32,
                "seed": 43  # Different seed
            })
            diff_codeword = encoder_diff_seed.encode(message)
            self.assertTrue(np.any(codeword != diff_codeword))


class TestDecoders(unittest.TestCase):
    """Test decoder implementations."""
    
    def test_bitwise_compare_decoder(self):
        """Test that the bitwise compare decoder works correctly."""
        # Create encoder and decoder
        encoder = HashTaggingEncoder({"code_length": 16})
        decoder = BitwiseCompareDecoder(encoder)
        
        # Generate a test message
        message = "test_message"
        wrong_message = "wrong_message"
        
        # Encode the message
        codeword = encoder.encode(message)
        
        # Test exact match (threshold = 1.0)
        decoder.set_parameters({"threshold": 1.0})
        self.assertTrue(decoder.decode(codeword, message))
        self.assertFalse(decoder.decode(codeword, wrong_message))
        
        # Test with lower threshold
        decoder.set_parameters({"threshold": 0.5})
        self.assertTrue(decoder.decode(codeword, message))
        
        # Test with noise in the codeword
        noisy_codeword = codeword.copy()
        noisy_codeword[0] = 1 - noisy_codeword[0]  # Flip one bit
        
        decoder.set_parameters({"threshold": 1.0})
        self.assertFalse(decoder.decode(noisy_codeword, message))
        
        decoder.set_parameters({"threshold": 0.9})
        self.assertTrue(decoder.decode(noisy_codeword, message))
    
    def test_hamming_distance_decoder(self):
        """Test that the Hamming distance decoder works correctly."""
        # Create encoder and decoder
        encoder = RandomProjectionEncoder({
            "code_length": 16,
            "seed": 42
        })
        decoder = HammingDistanceDecoder(encoder, {"max_distance": 0})
        
        # Generate a test message
        message = "test_message"
        wrong_message = "wrong_message"
        
        # Encode the message
        codeword = encoder.encode(message)
        
        # Test exact match (max_distance = 0)
        self.assertTrue(decoder.decode(codeword, message))
        self.assertFalse(decoder.decode(codeword, wrong_message))
        
        # Test with higher max_distance
        decoder.set_parameters({"max_distance": 2})
        
        # Test with noise in the codeword
        noisy_codeword = codeword.copy()
        noisy_codeword[0] = 1 - noisy_codeword[0]  # Flip one bit
        noisy_codeword[1] = 1 - noisy_codeword[1]  # Flip another bit
        
        self.assertTrue(decoder.decode(noisy_codeword, message))
        
        # Flip another bit, should now be rejected
        noisy_codeword[2] = 1 - noisy_codeword[2]
        self.assertFalse(decoder.decode(noisy_codeword, message))


class TestIdSystem(unittest.TestCase):
    """Test complete identification systems."""
    
    def test_hash_tagging_system(self):
        """Test that the hash tagging identification system works correctly."""
        # Create the system
        system = create_id_system("hash_tagging", {"code_length": 16})
        
        # Generate messages
        messages = generate_string_messages(10)
        
        for i, sender_msg in enumerate(messages):
            # Encode the message
            codeword = system.send(sender_msg)
            
            # Test identification with same message
            self.assertTrue(system.receive(codeword, sender_msg))
            
            # Test with other messages (should not match)
            for j, receiver_msg in enumerate(messages):
                if i != j:
                    self.assertFalse(system.receive(codeword, receiver_msg))
    
    def test_random_projection_system(self):
        """Test that the random projection identification system works correctly."""
        # Create the system with non-zero distance threshold
        system = create_id_system("random_projection", {
            "code_length": 16,
            "max_distance": 2,
            "seed": 42
        })
        
        # Generate messages
        messages = generate_string_messages(10)
        
        for i, sender_msg in enumerate(messages):
            # Encode the message
            codeword = system.send(sender_msg)
            
            # Test identification with same message
            self.assertTrue(system.receive(codeword, sender_msg))
            
            # Test with other messages (should not match)
            for j, receiver_msg in enumerate(messages):
                if i != j:
                    self.assertFalse(system.receive(codeword, receiver_msg))


class TestMetrics(unittest.TestCase):
    """Test metrics calculations."""
    
    def setUp(self):
        """Set up for metrics tests."""
        # Create a system and message set for testing
        self.system = create_id_system("hash_tagging", {"code_length": 8})
        self.message_set = generate_numeric_messages(20)
    
    def test_reliability(self):
        """Test reliability metric calculation."""
        # Calculate reliability
        reliability = IdMetrics.reliability(self.system, self.message_set, num_trials=100)
        
        # Check that reliability is a float between 0 and 1
        self.assertIsInstance(reliability, float)
        self.assertTrue(0 <= reliability <= 1)
        
        # With hash tagging and exact matching, reliability should be near 1
        self.assertGreater(reliability, 0.9)
    
    def test_error_rates(self):
        """Test error rate calculations."""
        # Calculate error rates
        error_rates = IdMetrics.error_rates(self.system, self.message_set, num_trials=100)
        
        # Check that all expected rates are present
        self.assertIn("false_positive_rate", error_rates)
        self.assertIn("false_negative_rate", error_rates)
        
        # Check that rates are floats between 0 and 1
        for rate in error_rates.values():
            self.assertIsInstance(rate, float)
            self.assertTrue(0 <= rate <= 1)
    
    def test_collision_matrix(self):
        """Test collision matrix calculation."""
        # Calculate collision matrix for a small set
        small_set = self.message_set[:5]
        collision_matrix = IdMetrics.collision_matrix(self.system, small_set)
        
        # Check matrix dimensions
        self.assertEqual(collision_matrix.shape, (5, 5))
        
        # Check that the matrix contains only boolean values
        self.assertTrue(np.all((collision_matrix == False) | (collision_matrix == True)))
        
        # Check that diagonal elements are not included (self-collisions)
        for i in range(5):
            self.assertFalse(collision_matrix[i, i])


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""
    
    def setUp(self):
        """Set up for visualization tests."""
        # Create a system and message set for testing
        self.system = create_id_system("hash_tagging", {"code_length": 8})
        self.message_set = generate_numeric_messages(20)
        plt.ion()  # Turn on interactive mode to avoid blocking
    
    def test_reliability_plot(self):
        """Test reliability vs code length plot."""
        # Create plot
        fig, ax = IdVisualizer.plot_reliability_vs_code_length(
            self.system, self.message_set, [4, 8, 12], num_trials=10
        )
        
        # Check that the plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Clean up
        plt.close(fig)
    
    def test_error_rates_plot(self):
        """Test error rates plot."""
        # Create plot
        fig, ax = IdVisualizer.plot_error_rates(
            self.system, self.message_set, [4, 8, 12], "code_length", num_trials=10
        )
        
        # Check that the plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Clean up
        plt.close(fig)
    
    def test_create_dashboard(self):
        """Test dashboard creation."""
        # Create dashboard
        fig = IdVisualizer.create_dashboard(
            self.system, self.message_set, [4, 8], num_trials=10
        )
        
        # Check that the dashboard was created
        self.assertIsNotNone(fig)
        
        # Clean up
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()