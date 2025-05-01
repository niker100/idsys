from abc import ABC, abstractmethod
import numpy as np
import hashlib
import warnings
import struct
import time
from typing import Any, Dict, List, Tuple, Union, Optional


class Encoder(ABC):
    """
    Abstract base class for all encoders in the identification system.
    """
    @abstractmethod
    def encode(self, message: Any) -> Any:
        """
        Encode a message according to the specific encoding scheme.
        
        Args:
            message: The message to encode
            
        Returns:
            The encoded message
        """
        pass
    
    @abstractmethod
    def get_code_rate(self) -> float:
        """
        Return the code rate (efficiency) of the encoder.
        
        Returns:
            float: The code rate
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters of the encoder.
        
        Returns:
            dict: Dictionary of encoder parameters
        """
        pass
    
    def analyze_performance(self, message_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze the theoretical performance of the encoder.
        
        Args:
            message_size: Size of test message in bytes
            
        Returns:
            dict: Performance metrics
        """
        # Default implementation - should be overridden by subclasses
        return {
            'encoder_type': self.__class__.__name__,
            'theoretical_analysis': 'Not implemented for this encoder type'
        }


class TaggingEncoder(Encoder):
    """
    Implementation of a tagging code encoder with configurable coding schemes and length.
    
    In tagging codes, the message is hashed or otherwise processed to create a tag of 
    specified length, which serves as the compressed representation for identification.
    
    Key advantages over traditional Shannon-based communication:
    1. Bandwidth efficiency: Only need to transmit the tag, not the full message
    2. Constant size: Tags have fixed size regardless of message length
    3. Fast verification: Simple equality check rather than full decoding
    
    Limitations:
    1. Collisions: Different messages may map to the same tag
    2. No error correction: Most tagging schemes can't recover from transmission errors
    3. One-way: Original message can't be recovered from the tag
    """
    def __init__(self, tag_length: int = 32, coding_scheme: str = 'hash', 
                 error_tolerance: float = 0.0):
        """
        Initialize the tagging encoder.
        
        Args:
            tag_length (int): Length of the tag in bits (1-128)
            coding_scheme (str): Coding scheme to use ('hash', 'truncate', 'mod', 'locality_sensitive')
            error_tolerance (float): Error tolerance level (0.0-1.0), for locality-sensitive schemes
        """
        # Validate inputs
        if not isinstance(tag_length, int):
            raise TypeError("tag_length must be an integer")
        if tag_length <= 0:
            raise ValueError("tag_length must be positive")
        if tag_length > 128:
            warnings.warn(f"tag_length > 128 bits is excessive for most applications. Using 128 bits.")
            self.tag_length = 128
        else:
            self.tag_length = tag_length
            
        valid_schemes = ['hash', 'truncate', 'mod', 'locality_sensitive']
        if coding_scheme not in valid_schemes:
            raise ValueError(f"coding_scheme must be one of {valid_schemes}")
        self.coding_scheme = coding_scheme
        
        # Scale warning based on application size
        if tag_length < 16 and coding_scheme == 'hash':
            warnings.warn(
                f"Using {tag_length} bits with hash coding may result in high collision rates "
                f"for real-world applications. For context: 16 bits = 1 in 65k collision chance "
                f"with just 300 messages.")
            
        # For real-world applications:
        # - 32-bit tags: suitable for hundreds of thousands of items
        # - 64-bit tags: suitable for billions of items
        # - 128-bit tags: suitable for astronomical scale (effectively collision-free)
        
        # Set error tolerance for locality-sensitive hashing
        self.error_tolerance = max(0.0, min(1.0, error_tolerance))
        
        # Pre-calculate max tag value for efficiency
        self._max_tag_value = (1 << self.tag_length) - 1
        
        # Performance counters
        self._encode_calls = 0
        self._encode_time_total = 0
        
    def encode(self, message: Any) -> int:
        """
        Encode a message using the specified tagging scheme.
        
        Args:
            message: The message to encode
            
        Returns:
            The encoded tag (integer representation)
        """
        # Track performance
        start_time = time.time()
        self._encode_calls += 1
        
        # Convert message to bytes if not already
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        elif isinstance(message, (int, float)):
            message_bytes = str(message).encode('utf-8')
        elif isinstance(message, bytes):
            message_bytes = message
        else:
            # Convert other types to string representation
            message_bytes = str(message).encode('utf-8')
        
        result = self._do_encode(message_bytes)
        
        # Update timing stats
        self._encode_time_total += time.time() - start_time
        
        return result
    
    def _do_encode(self, message_bytes: bytes) -> int:
        """Internal method to perform encoding based on selected scheme"""
        if self.coding_scheme == 'hash':
            # Use a cryptographic hash function (SHA-256)
            hash_full = hashlib.sha256(message_bytes).digest()
            # Truncate to desired tag length (in bits)
            return int.from_bytes(hash_full, byteorder='big') & self._max_tag_value
            
        elif self.coding_scheme == 'truncate':
            # Simple truncation of the message bytes
            if len(message_bytes) > 0:
                # Take first bytes needed and convert to integer
                num_bytes = (self.tag_length + 7) // 8  # Ceiling division
                truncated = message_bytes[:num_bytes]
                # Pad with zeros if needed
                if len(truncated) < num_bytes:
                    truncated = truncated + b'\x00' * (num_bytes - len(truncated))
                # Convert to integer and mask to correct bit length
                numeric_val = int.from_bytes(truncated, byteorder='big')
                return numeric_val & self._max_tag_value
            else:
                return 0
            
        elif self.coding_scheme == 'mod':
            # Modulo-based scheme - suitable for numeric data
            if isinstance(message_bytes, bytes):
                # Convert bytes to large integer and take modulo
                numeric_val = int.from_bytes(message_bytes, byteorder='big')
            else:
                numeric_val = int(message_bytes)
            
            return numeric_val % (2 ** self.tag_length)
            
        elif self.coding_scheme == 'locality_sensitive':
            # Locality-sensitive hashing - can tolerate minor differences
            # Implement a simple version of SimHash
            # This allows approximate matching, useful for fuzzy identification
            
            # Convert to bytes and ensure sufficient length
            if len(message_bytes) < 4:  # Need at least 4 bytes for meaningful LSH
                message_bytes = message_bytes + message_bytes * (4 - len(message_bytes))
            
            # Create a feature vector from the message
            # For simplicity, we use overlapping 4-byte windows
            features = [message_bytes[i:i+4] for i in range(len(message_bytes) - 3)]
            
            # Initialize a vector of tag_length dimensions
            v = np.zeros(self.tag_length, dtype=float)
            
            # Update vector based on features
            for feature in features:
                # Use the feature to seed a hash function
                h = hashlib.md5(feature).digest()
                # Use each bit to update one dimension of our vector
                for i in range(min(len(h), self.tag_length)):
                    bit = (h[i >> 3] >> (i & 7)) & 1
                    v[i] += 1 if bit else -1
            
            # Convert to binary tag
            tag = 0
            for i in range(self.tag_length):
                if v[i] > 0:
                    tag |= (1 << i)
            
            return tag
            
        # Should never reach here due to validation in __init__
        raise ValueError(f"Unknown coding scheme: {self.coding_scheme}")
    
    def similarity(self, tag1: int, tag2: int) -> float:
        """
        Calculate similarity between two tags (only meaningful for locality-sensitive hashing).
        
        Args:
            tag1: First tag
            tag2: Second tag
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if self.coding_scheme != 'locality_sensitive':
            # For non-LSH schemes, tags are either identical or different
            return 1.0 if tag1 == tag2 else 0.0
            
        # For LSH, calculate Hamming similarity
        xor_result = tag1 ^ tag2
        hamming_distance = bin(xor_result).count('1')
        return 1.0 - (hamming_distance / self.tag_length)
    
    def is_similar(self, tag1: int, tag2: int) -> bool:
        """
        Determine if two tags are similar based on the error tolerance.
        
        Args:
            tag1: First tag
            tag2: Second tag
            
        Returns:
            bool: True if tags are similar enough based on error_tolerance
        """
        sim = self.similarity(tag1, tag2)
        return sim >= (1.0 - self.error_tolerance)
    
    def get_code_rate(self) -> float:
        """
        Return the code rate for this tagging encoder.
        For tagging codes with arbitrary message lengths, this approaches 0
        as message length grows, meaning high compression.
        
        Returns:
            float: The code rate (bits of tag / bits of original message)
        """
        return self.tag_length / 8  # Approximate bits per byte of message
    
    def get_collision_probability(self, num_messages: int) -> float:
        """
        Calculate the theoretical probability of at least one collision
        when encoding num_messages distinct messages with this encoder.
        
        Args:
            num_messages: Number of distinct messages to encode
            
        Returns:
            float: Probability of at least one collision
        """
        m = 2 ** self.tag_length  # Number of possible tags
        n = num_messages           # Number of messages
        
        if n > m:
            return 1.0  # Collision guaranteed by pigeonhole principle
        
        if n <= 1:
            return 0.0  # No collision possible with 0 or 1 message
        
        # Use the birthday problem formula
        # For large m, approximated as: P(collision) ≈ 1 - exp(-n(n-1)/2m)
        return 1 - np.exp(-n * (n-1) / (2 * m))
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters of this encoder.
        
        Returns:
            dict: Dictionary of encoder parameters
        """
        avg_time_per_encode = 0
        if self._encode_calls > 0:
            avg_time_per_encode = self._encode_time_total / self._encode_calls
            
        return {
            'tag_length': self.tag_length,
            'coding_scheme': self.coding_scheme,
            'max_tag_value': self._max_tag_value,
            'distinct_tags_possible': 2 ** self.tag_length,
            'error_tolerance': self.error_tolerance,
            'encode_calls': self._encode_calls,
            'avg_encoding_time_ms': avg_time_per_encode * 1000,
        }
    
    def analyze_performance(self, message_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze the theoretical performance of the tagging encoder.
        
        Args:
            message_size: Size of test message in bytes
            
        Returns:
            dict: Performance analysis
        """
        # Common collision probability scales
        collision_probs = {
            '100_msgs': self.get_collision_probability(100),
            '10K_msgs': self.get_collision_probability(10_000),
            '1M_msgs': self.get_collision_probability(1_000_000),
            '1B_msgs': self.get_collision_probability(1_000_000_000),
        }
        
        # Calculate message-to-tag compression ratio
        compression_ratio = message_size / ((self.tag_length + 7) // 8)
        
        # Bandwidth savings compared to Shannon-bounded communication
        # Assuming Shannon capacity of 2 bits per symbol (moderate SNR)
        shannon_bits = message_size * 8
        tagging_bits = self.tag_length
        bandwidth_saving = shannon_bits / tagging_bits if tagging_bits > 0 else float('inf')
        
        # Reliability analysis
        if self.coding_scheme == 'locality_sensitive':
            # LSH has inherent error tolerance
            error_tolerance_rating = "Moderate - can handle minor differences"
            max_error_rate = self.error_tolerance
        else:
            # Other schemes require exact match
            error_tolerance_rating = "Low - requires exact match"
            max_error_rate = 0.0
        
        # Real-world scale applicability
        scale_rating = "Unknown"
        if self.tag_length < 16:
            scale_rating = "Very small applications only (<1K items)"
        elif self.tag_length < 32:
            scale_rating = "Small applications (1K-100K items)"
        elif self.tag_length < 64:
            scale_rating = "Medium applications (100K-10M items)"
        elif self.tag_length < 96:
            scale_rating = "Large applications (10M-1B items)"
        else:
            scale_rating = "Enterprise/global scale (>1B items)"
            
        # Security level (comparable to symmetric key strength)
        security_level = min(self.tag_length, self.tag_length / 2 if self.coding_scheme != 'hash' else self.tag_length)
        
        return {
            'encoder_type': self.__class__.__name__,
            'coding_scheme': self.coding_scheme,
            'tag_length_bits': self.tag_length,
            'compression_ratio': compression_ratio,
            'bandwidth_saving_vs_shannon': bandwidth_saving,
            'collision_probabilities': collision_probs,
            'error_tolerance': error_tolerance_rating,
            'max_bit_error_rate': max_error_rate,
            'real_world_scale': scale_rating,
            'security_level_bits': security_level,
            'limitations': self._get_limitations(),
            'advantages': self._get_advantages(),
        }
        
    def _get_limitations(self) -> List[str]:
        """Identify the limitations of the current configuration"""
        limitations = []
        
        # Collision limitations based on tag length
        if self.tag_length < 16:
            limitations.append("High collision rate even with few messages")
        elif self.tag_length < 32:
            limitations.append("Moderate collision risk with thousands of messages")
        
        # Scheme-specific limitations
        if self.coding_scheme == 'hash':
            limitations.append("Non-recoverable: original message cannot be reconstructed from tag")
            limitations.append("No distance preservation: similar messages may have very different tags")
        elif self.coding_scheme == 'truncate':
            limitations.append("Low entropy utilization: only uses first few bytes of message")
            limitations.append("Poor distribution for structured data")
        elif self.coding_scheme == 'mod':
            limitations.append("Poor distribution with non-uniform message values")
            limitations.append("Weak security properties")
        elif self.coding_scheme == 'locality_sensitive':
            limitations.append("Reduced discrimination power compared to cryptographic hashing")
            limitations.append("May produce false positives for similar but distinct messages")
        
        # Common limitations for all tagging codes
        limitations.append("Cannot correct transmission errors without additional error coding")
        
        return limitations
    
    def _get_advantages(self) -> List[str]:
        """Identify the advantages of the current configuration"""
        advantages = []
        
        # Common advantages for all tagging codes
        advantages.append("Fixed-size output regardless of input size")
        advantages.append(f"Bandwidth efficient: {self.tag_length} bits vs full message transmission")
        advantages.append("Fast verification through simple equality check")
        
        # Scheme-specific advantages
        if self.coding_scheme == 'hash':
            advantages.append("Strong uniformity in tag distribution")
            advantages.append("Cryptographic security properties")
        elif self.coding_scheme == 'truncate':
            advantages.append("Extremely fast encoding")
            advantages.append("Simple implementation")
        elif self.coding_scheme == 'mod':
            advantages.append("Fast encoding even for large messages")
            advantages.append("Suitable for numeric data")
        elif self.coding_scheme == 'locality_sensitive':
            advantages.append(f"Tolerates message differences up to {self.error_tolerance * 100:.1f}%")
            advantages.append("Can identify similar messages, not just exact matches")
        
        # Scale-dependent advantages
        if self.tag_length >= 64:
            advantages.append("Practically collision-free for most real-world applications")
        
        return advantages
    
    def __str__(self) -> str:
        """Return a string representation of the encoder"""
        return f"TaggingEncoder(tag_length={self.tag_length}, coding_scheme='{self.coding_scheme}')"


class RealWorldTaggingEncoder(TaggingEncoder):
    """
    Extension of TaggingEncoder optimized for real-world applications with
    standardized sizes and practical collision resistance.
    
    Provides pre-configured options for common real-world use cases.
    """
    
    # Standard tag size configurations for different scales
    SCALE_CONFIGS = {
        'small': {'tag_length': 32, 'items': '< 65,000', 'collision_prob_1k': 7.6e-6},
        'medium': {'tag_length': 64, 'items': '< 5 million', 'collision_prob_1m': 4.3e-7},
        'large': {'tag_length': 96, 'items': '< 1 billion', 'collision_prob_1b': 1.7e-10},
        'enterprise': {'tag_length': 128, 'items': 'effectively infinite', 'collision_prob_1b': 2.0e-23}
    }
    
    def __init__(self, scale: str = 'medium', coding_scheme: str = 'hash', 
                 error_tolerant: bool = False):
        """
        Initialize with a predefined scale configuration.
        
        Args:
            scale (str): Scale of application - 'small', 'medium', 'large', or 'enterprise'
            coding_scheme (str): Coding scheme to use ('hash', 'truncate', 'mod', 'locality_sensitive')
            error_tolerant (bool): Whether to enable error tolerance
        """
        if scale not in self.SCALE_CONFIGS:
            raise ValueError(f"Scale must be one of {list(self.SCALE_CONFIGS.keys())}")
        
        # Set tag length based on scale
        tag_length = self.SCALE_CONFIGS[scale]['tag_length']
        
        # Set error tolerance if requested
        error_tolerance = 0.1 if error_tolerant else 0.0
        
        # Use LSH coding scheme if error tolerant
        final_scheme = 'locality_sensitive' if error_tolerant else coding_scheme
        
        # Initialize the parent class
        super().__init__(tag_length=tag_length, coding_scheme=final_scheme, 
                         error_tolerance=error_tolerance)
        
        # Store additional information
        self.scale = scale
        self.scale_info = self.SCALE_CONFIGS[scale]
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters of this encoder.
        
        Returns:
            dict: Dictionary of encoder parameters
        """
        params = super().get_parameters()
        params.update({
            'scale': self.scale,
            'recommended_items': self.scale_info['items'],
        })
        return params
    
    def __str__(self) -> str:
        """Return a string representation of the encoder"""
        return f"RealWorldTaggingEncoder(scale='{self.scale}', coding_scheme='{self.coding_scheme}')"


class ErrorDetectingEncoder(Encoder):
    """
    Encoder that adds error detection capability to the identification system.
    
    Instead of just mapping a message to a tag, this encoder also generates
    an error detection code that can be used to verify message integrity.
    """
    
    def __init__(self, method: str = 'crc32', encoder: Optional[Encoder] = None):
        """
        Initialize the error detecting encoder.
        
        Args:
            method (str): Error detection method: 'parity', 'crc8', 'crc16', 'crc32'
            encoder (Optional[Encoder]): Base encoder to use (if None, uses TaggingEncoder)
        """
        valid_methods = ['parity', 'crc8', 'crc16', 'crc32']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        self.method = method
        
        # Determine number of bits used for error detection
        if method == 'parity':
            self.bits = 1
        elif method == 'crc8':
            self.bits = 8
        elif method == 'crc16':
            self.bits = 16
        elif method == 'crc32':
            self.bits = 32
            
        # Use default encoder if none provided
        self.base_encoder = encoder if encoder is not None else TaggingEncoder(32)
    
    def encode(self, message: Any) -> Tuple[Any, int]:
        """
        Encode a message and add error detection.
        
        Args:
            message: The message to encode
            
        Returns:
            tuple: (encoded_message, error_code)
        """
        # First encode the message using the base encoder
        encoded = self.base_encoder.encode(message)
        
        # Convert message to bytes for error code calculation
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        elif isinstance(message, (int, float)):
            message_bytes = str(message).encode('utf-8')
        elif isinstance(message, bytes):
            message_bytes = message
        else:
            message_bytes = str(message).encode('utf-8')
        
        # Compute error detection code
        if self.method == 'parity':
            # Simple parity bit (count of 1 bits)
            ones_count = 0
            for b in message_bytes:
                ones_count += bin(b).count('1')
            error_code = ones_count % 2
            
        elif self.method.startswith('crc'):
            import zlib
            
            if self.method == 'crc32':
                error_code = zlib.crc32(message_bytes) & 0xffffffff
            elif self.method == 'crc16':
                # Use CRC32 and truncate to 16 bits
                error_code = zlib.crc32(message_bytes) & 0xffff
            elif self.method == 'crc8':
                # Use CRC32 and truncate to 8 bits
                error_code = zlib.crc32(message_bytes) & 0xff
        
        return (encoded, error_code)
    
    def verify(self, message: Any, error_code: int) -> bool:
        """
        Verify message integrity using the error code.
        
        Args:
            message: The original message
            error_code: The error code to check against
            
        Returns:
            bool: True if verification passes
        """
        # Re-compute the error code and compare
        _, new_code = self.encode(message)
        return error_code == new_code
    
    def get_code_rate(self) -> float:
        """
        Return the code rate for this encoder.
        
        Returns:
            float: Code rate
        """
        # Code rate is k/(k+r) where k is message bits and r is redundancy bits
        # For error detection, we add a fixed number of bits, so rate approaches 1
        # for large messages
        return self.base_encoder.get_code_rate() * 0.9  # Approximate penalty for error detection
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters of the encoder.
        
        Returns:
            dict: Dictionary of encoder parameters
        """
        base_params = self.base_encoder.get_parameters()
        return {
            'error_detection_method': self.method,
            'error_detection_bits': self.bits,
            'base_encoder': self.base_encoder.__class__.__name__,
            'base_encoder_params': base_params
        }
    
    def analyze_performance(self, message_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze the theoretical performance of the error detecting encoder.
        
        Args:
            message_size: Size of test message in bytes
            
        Returns:
            dict: Performance analysis
        """
        base_analysis = self.base_encoder.analyze_performance(message_size)
        
        # Calculate error detection capability
        undetected_error_rate = 0.0
        if self.method == 'parity':
            # Parity can detect odd number of bit errors
            undetected_error_rate = 0.5  # Misses 50% of possible error patterns
        elif self.method == 'crc8':
            # CRC-8 can detect all 1-bit errors and most burst errors up to 8 bits
            undetected_error_rate = 1.0 / (2**8)
        elif self.method == 'crc16':
            # CRC-16 can detect all 1-bit errors and most burst errors up to 16 bits
            undetected_error_rate = 1.0 / (2**16)
        elif self.method == 'crc32':
            # CRC-32 can detect all 1-bit errors and most burst errors up to 32 bits
            undetected_error_rate = 1.0 / (2**32)
        
        # Calculate overhead
        overhead_bits = self.bits
        overhead_ratio = overhead_bits / (message_size * 8)
        
        # Add error detection specific metrics
        error_metrics = {
            'undetected_error_rate': undetected_error_rate,
            'error_detection_bits': self.bits,
            'overhead_ratio': overhead_ratio,
            'advantages': [
                f"Can detect transmission errors with {(1 - undetected_error_rate) * 100:.6f}% reliability",
                "Prevents false positives from corruption during transmission",
                f"Adds only {overhead_ratio * 100:.2f}% overhead for error detection"
            ],
            'limitations': [
                "Cannot correct errors, only detect them",
                "Increases message size slightly",
                "Adds computational overhead for error code calculation"
            ]
        }
        
        # Combine with base analysis
        combined = base_analysis.copy()
        combined.update(error_metrics)
        combined['encoder_type'] = f"{self.__class__.__name__}+{self.base_encoder.__class__.__name__}"
        
        return combined
        
    def __str__(self) -> str:
        """Return a string representation of the encoder"""
        return f"ErrorDetectingEncoder(method='{self.method}', base_encoder={self.base_encoder})"