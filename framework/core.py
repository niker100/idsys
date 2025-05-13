#!/usr/bin/env python3
"""
Core module for identification systems.

This module provides the fundamental classes and functions for
implementing and evaluating identification systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import hashlib
import random


class IdEncoder:
    """Base class for identification system encoders."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the encoder with parameters.
        
        Args:
            parameters: Dictionary of encoder parameters
        """
        self.parameters = parameters or {}
        
    def encode(self, message: Any) -> Any:
        """
        Encode a message.
        
        Args:
            message: The message to encode
            
        Returns:
            The encoded message (codeword)
        """
        raise NotImplementedError("Subclasses must implement encode()")
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update encoder parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.parameters.update(parameters)


class IdDecoder:
    """Base class for identification system decoders."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the decoder with parameters.
        
        Args:
            parameters: Dictionary of decoder parameters
        """
        self.parameters = parameters or {}
        
    def decode(self, codeword: Any, message: Any) -> bool:
        """
        Decide whether the given message matches the codeword.
        
        Args:
            codeword: The received codeword
            message: The message to check
            
        Returns:
            True if identified as matching, False otherwise
        """
        raise NotImplementedError("Subclasses must implement decode()")
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update decoder parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.parameters.update(parameters)


class IdSystem:
    """Complete identification system combining encoder and decoder."""
    
    def __init__(self, encoder: IdEncoder, decoder: IdDecoder):
        """
        Initialize the identification system.
        
        Args:
            encoder: The encoder to use
            decoder: The decoder to use
        """
        self.encoder = encoder
        self.decoder = decoder
        
    def send(self, message: Any) -> Any:
        """
        Encode and send a message.
        
        Args:
            message: The message to send
            
        Returns:
            The encoded message (codeword)
        """
        return self.encoder.encode(message)
        
    def receive(self, codeword: Any, message: Any) -> bool:
        """
        Decide if the received codeword matches the message.
        
        Args:
            codeword: The received codeword
            message: The message to check
            
        Returns:
            True if identified as matching, False otherwise
        """
        return self.decoder.decode(codeword, message)


class HashTaggingEncoder(IdEncoder):
    """
    Encoder that uses cryptographic hash functions for tagging messages.
    
    This implements a basic identification coding scheme using hashes.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the hash tagging encoder.
        
        Args:
            parameters: Dictionary with parameters like:
                - code_length: Length of the hash output in bits
                - hash_function: Which hash function to use ('sha256', 'md5', etc.)
        """
        default_params = {
            "code_length": 8,  # Default to 8 bits
            "hash_function": "sha256"
        }
        
        # Set default parameters, then update with any provided parameters
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
            
    def encode(self, message: Any) -> np.ndarray:
        """
        Encode a message using hash-based tagging.
        
        Args:
            message: The message to encode
            
        Returns:
            np.ndarray: Binary array representing the tag
        """
        # Convert message to string if it isn't already
        if not isinstance(message, str):
            message = str(message)
            
        # Create hash of the message
        if self.parameters["hash_function"] == "md5":
            hash_obj = hashlib.md5(message.encode())
        else:
            # Default to SHA-256
            hash_obj = hashlib.sha256(message.encode())
            
        # Get the hex digest and convert to binary
        hex_digest = hash_obj.hexdigest()
        
        # Convert hex to binary representation
        binary = bin(int(hex_digest, 16))[2:]  # Remove '0b' prefix
        
        # Ensure we have enough bits
        binary = binary.zfill(256)  # SHA-256 gives 256 bits
        
        # Take the first code_length bits
        code_length = self.parameters["code_length"]
        binary = binary[:code_length]
        
        # Convert to binary array
        binary_array = np.array([int(bit) for bit in binary], dtype=np.int8)
        
        return binary_array


class BitwiseCompareDecoder(IdDecoder):
    """
    Decoder that compares codewords bitwise.
    
    This implements a simple bitwise comparison between the received codeword
    and the locally computed codeword for the message being checked.
    """
    
    def __init__(self, encoder: IdEncoder, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the bitwise compare decoder.
        
        Args:
            encoder: The encoder used by the sender (for local encoding)
            parameters: Dictionary with parameters like:
                - threshold: Percentage of bits that must match (0.0 to 1.0)
        """
        default_params = {
            "threshold": 1.0  # Default to exact match
        }
        
        # Set default parameters, then update with any provided parameters
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
            
        self.encoder = encoder
            
    def decode(self, received_codeword: np.ndarray, message: Any) -> bool:
        """
        Decide if the received codeword matches the message.
        
        Args:
            received_codeword: The received codeword
            message: The message to check
            
        Returns:
            True if identified as matching, False otherwise
        """
        # Encode the message using the same encoder
        local_codeword = self.encoder.encode(message)
        
        # Compare the codewords
        if len(received_codeword) != len(local_codeword):
            return False
        
        # Calculate the percentage of matching bits
        matches = np.sum(received_codeword == local_codeword)
        match_percentage = matches / len(received_codeword)
        
        # Compare with threshold
        return match_percentage >= self.parameters["threshold"]


class RandomProjectionEncoder(IdEncoder):
    """
    Encoder that uses random projections for identification.
    
    This implements a more advanced identification coding scheme using
    random projections, which can be useful for approximate matching.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the random projection encoder.
        
        Args:
            parameters: Dictionary with parameters like:
                - code_length: Number of dimensions in the output
                - feature_size: Size of the feature vector
                - seed: Random seed for reproducibility
        """
        default_params = {
            "code_length": 8,      # Default output dimensions
            "feature_size": 32,    # Default feature size
            "seed": None           # Random seed
        }
        
        # Set default parameters, then update with any provided parameters
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
            
        # Initialize random projection matrix
        self._initialize_projection_matrix()
            
    def _initialize_projection_matrix(self):
        """Initialize the random projection matrix."""
        code_length = self.parameters["code_length"]
        feature_size = self.parameters["feature_size"]
        
        # Set random seed for reproducibility if specified
        if self.parameters["seed"] is not None:
            np.random.seed(self.parameters["seed"])
            
        # Create random projection matrix
        self.projection_matrix = np.random.randn(feature_size, code_length)
    
    def _extract_features(self, message: Any) -> np.ndarray:
        """
        Extract feature vector from a message.
        
        Args:
            message: The message to encode
            
        Returns:
            Feature vector as numpy array
        """
        # Convert message to string if it isn't already
        if not isinstance(message, str):
            message = str(message)
            
        # Simple feature extraction: Use hash value
        hash_obj = hashlib.sha256(message.encode())
        hex_digest = hash_obj.hexdigest()
        
        # Convert to feature vector (normalize each byte to [-1, 1])
        feature_size = self.parameters["feature_size"]
        hex_bytes = bytes.fromhex(hex_digest)
        features = np.array([int(b) for b in hex_bytes[:feature_size]]) / 128.0 - 1.0
        
        # Ensure proper size
        if len(features) < feature_size:
            features = np.pad(features, (0, feature_size - len(features)))
            
        return features
            
    def encode(self, message: Any) -> np.ndarray:
        """
        Encode a message using random projection.
        
        Args:
            message: The message to encode
            
        Returns:
            np.ndarray: Binary array representing the code
        """
        # Extract features
        features = self._extract_features(message)
        
        # Apply random projection
        projection = features @ self.projection_matrix
        
        # Binarize the projection
        binary_code = (projection > 0).astype(np.int8)
        
        return binary_code
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update encoder parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        super().set_parameters(parameters)
        
        # If code_length or feature_size changed, reinitialize the projection matrix
        if "code_length" in parameters or "feature_size" in parameters:
            self._initialize_projection_matrix()


class HammingDistanceDecoder(IdDecoder):
    """
    Decoder that uses Hamming distance for identification.
    
    This implements a decoder based on Hamming distance, which counts the
    number of differing bits between codewords.
    """
    
    def __init__(self, encoder: IdEncoder, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hamming distance decoder.
        
        Args:
            encoder: The encoder used by the sender (for local encoding)
            parameters: Dictionary with parameters like:
                - max_distance: Maximum Hamming distance allowed for identification
        """
        default_params = {
            "max_distance": 0  # Default to exact match
        }
        
        # Set default parameters, then update with any provided parameters
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
            
        self.encoder = encoder
            
    def decode(self, received_codeword: np.ndarray, message: Any) -> bool:
        """
        Decide if the received codeword matches the message.
        
        Args:
            received_codeword: The received codeword
            message: The message to check
            
        Returns:
            True if identified as matching, False otherwise
        """
        # Encode the message using the same encoder
        local_codeword = self.encoder.encode(message)
        
        # Compare the codewords using Hamming distance
        if len(received_codeword) != len(local_codeword):
            return False
        
        # Calculate Hamming distance (count of different bits)
        distance = np.sum(received_codeword != local_codeword)
        
        # Compare with threshold
        return distance <= self.parameters["max_distance"]


def create_id_system(system_type: str = "hash_tagging", parameters: Optional[Dict[str, Any]] = None) -> IdSystem:
    """
    Factory function to create an identification system.
    
    Args:
        system_type: Type of system to create ('hash_tagging' or 'random_projection')
        parameters: Dictionary of parameters for the system
        
    Returns:
        IdSystem: Configured identification system
    """
    parameters = parameters or {}
    
    if system_type == "hash_tagging":
        encoder = HashTaggingEncoder(parameters)
        decoder = BitwiseCompareDecoder(encoder, parameters)
    elif system_type == "random_projection":
        encoder = RandomProjectionEncoder(parameters)
        decoder = HammingDistanceDecoder(encoder, {"max_distance": parameters.get("max_distance", 0)})
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    return IdSystem(encoder, decoder)


# Message generation utilities

def generate_numeric_messages(count: int, min_value: int = 0, max_value: int = 1000000) -> List[int]:
    """
    Generate a list of random numeric messages.
    
    Args:
        count: Number of messages to generate
        min_value: Minimum value for messages
        max_value: Maximum value for messages
        
    Returns:
        List of random integers
    """
    return [random.randint(min_value, max_value) for _ in range(count)]


def generate_string_messages(count: int, length: int = 10) -> List[str]:
    """
    Generate a list of random string messages.
    
    Args:
        count: Number of messages to generate
        length: Length of each message
        
    Returns:
        List of random strings
    """
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return [''.join(random.choice(chars) for _ in range(length)) for _ in range(count)]