#!/usr/bin/env python3
"""
Core module for identification systems using the idcodes library.

This module provides the fundamental classes and functions for implementing and evaluating
identification systems with multiple coding schemes including Reed-Solomon (RSID),
concatenated Reed-Solomon (RS2ID), Reed-Muller (RMID), and hash-based systems (SHA1ID, SHA256ID).

The module follows a clean architecture pattern with separate encoder and verifier classes
for each identification system type, unified under a common IdSystem interface.

Copyright (c) 2025 Nick Schubert
Licensed under the MIT License - see LICENSE file for details

Classes:
    IdEncoder: Abstract base class for all encoders
    IdVerifier: Abstract base class for all verifiers  
    IdSystem: Main system interface combining encoder and verifier
    RSIDEncoder/RSIDVerifier: Reed-Solomon identification system
    RS2IDEncoder/RS2IDVerifier: Concatenated Reed-Solomon identification system  
    RMIDEncoder/RMIDVerifier: Reed-Muller identification system
    SHA1IDEncoder/SHA1IDVerifier: SHA1-based identification system
    SHA256IDEncoder/SHA256IDVerifier: SHA256-based identification system
    NoCodeEncoder/NoCodeVerifier: Baseline system for comparison

Functions:
    create_id_system: Factory function to create identification systems

Example:
    >>> system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
    >>> message = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> tag = system.send(message)
    >>> is_valid = system.receive(tag, message)
    >>> print(is_valid)  # True
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from .common import *


class IdEncoder(ABC):
    """
    Abstract base class for identification system encoders.
    
    All encoder implementations must inherit from this class and implement the encode method.
    Encoders are responsible for generating identification tags from input messages.
    
    Attributes:
        parameters: Dictionary containing encoder-specific parameters such as 
                   Galois field exponent, tag positions, etc.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """
        Initialize the encoder with optional parameters.
        
        Args:
            parameters: Dictionary of encoder parameters. Each encoder type
                       has its own set of required and optional parameters.
        """
        self.parameters: Parameters = parameters or {}

    @abstractmethod
    def encode(self, message: Message) -> Tag:
        """
        Encode a message to generate an identification tag.
        
        Args:
            message: Input message as a list of integers
            
        Returns:
            Identification tag(s) - can be single integer or list of integers
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement encode method")

    def set_parameters(self, parameters: Parameters) -> None:
        """
        Update encoder parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.parameters.update(parameters)


class IdVerifier(ABC):
    """
    Abstract base class for identification system verifiers.
    
    All verifier implementations must inherit from this class and implement the verify method.
    Verifiers check whether a given tag corresponds to a specific message.
    
    Attributes:
        parameters: Dictionary containing verifier-specific parameters
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """
        Initialize the verifier with optional parameters.
        
        Args:
            parameters: Dictionary of verifier parameters
        """
        self.parameters: Parameters = parameters or {}

    @abstractmethod
    def verify(self, codeword: Tag, message: Message) -> bool:
        """
        Verify if a codeword corresponds to a message.
        
        Args:
            codeword: The identification tag to verify
            message: The message to check against
            
        Returns:
            True if the codeword is valid for the message, False otherwise
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement verify method")

    def set_parameters(self, parameters: Parameters) -> None:
        """
        Update verifier parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.parameters.update(parameters)


class IdSystem:
    """
    Main identification system that combines an encoder and verifier.
    
    This class provides a unified interface for identification systems,
    abstracting away the details of specific encoder/verifier implementations.
    
    Attributes:
        encoder: The encoder instance used for generating tags
        verifier: The verifier instance used for verification
    """
    
    def __init__(self, encoder: IdEncoder, verifier: IdVerifier) -> None:
        """
        Initialize the identification system.
        
        Args:
            encoder: Encoder instance for tag generation
            verifier: Verifier instance for tag verification
        """
        self.encoder = encoder
        self.verifier = verifier

    def send(self, message: Message) -> Tag:
        """
        Generate an identification tag for a message (encoding phase).
        
        Args:
            message: Input message as list of integers
            
        Returns:
            Identification tag(s)
        """
        return self.encoder.encode(message)

    def receive(self, codeword: Tag, message: Message) -> bool:
        """
        Verify if a codeword corresponds to a message (verification phase).
        
        Args:
            codeword: The identification tag to verify
            message: The message to check against
            
        Returns:
            True if verification succeeds, False otherwise
        """
        return self.verifier.verify(codeword, message)
    
    def receive_k(self, codeword: Tag, messages: List[Message]) -> Optional[Message]:
        """
        Verify a codeword against multiple messages (k-verification).
        
        This method checks if any of the provided messages matches the given codeword.
        It's useful for false positive testing and collision detection.
        
        Args:
            codeword: The identification tag to verify against
            messages: List of messages to check
            
        Returns:
            The first message that matches the codeword, or None if no match is found
        """
        for msg in messages:
            if self.verifier.verify(codeword, msg):
                return msg
        return None


class RSIDEncoder(IdEncoder):
    """
    Reed-Solomon Identification (RSID) encoder.
    
    RSID is based on Reed-Solomon error-correcting codes and provides strong
    identification capabilities with configurable redundancy levels.
    
    Parameters:
        gf_exp: Galois field exponent (default: 8)
        tag_pos: List of tag positions for multi-tag systems (default: [2])
    
    Example:
        >>> encoder = RSIDEncoder({"gf_exp": 8, "tag_pos": [2, 3]})
        >>> message = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> tags = encoder.encode(message)
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RSID encoder with default parameters."""
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2]
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize the idcodes library instance for RSID operations."""
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = get_idcodes_instance(self.gf_exp)
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters and reinitialize idcodes if necessary."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> List[int]:
        """
        Encode a message using Reed-Solomon identification.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            List of identification tags (one per tag position)
            
        Raises:
            ValueError: If message is empty or parameters are invalid
        """
        if not message:
            raise ValueError("Message cannot be empty")
            
        tags = []
        for tag_pos in self.parameters["tag_pos"]:
            tags.append(self.idcodes.rsid(message, tag_pos, self.gf_exp))
        return tags


class RSIDVerifier(IdVerifier):
    """
    Reed-Solomon Identification (RSID) verifier.
    
    Verifies RSID tags by recomputing them and comparing with the provided codewords.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RSID verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = RSIDEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codewords: List[int], message: Message) -> bool:
        """
        Verify RSID codewords against a message.
        
        Args:
            codewords: List of identification tags to verify
            message: Message to check against
            
        Returns:
            True if all codewords match the recomputed tags
        """
        if not message:
            return False
            
        recomputed_tags = self.encoder.encode(message)
        return all(tag == codewords[i] for i, tag in enumerate(recomputed_tags))

class RS2IDEncoder(IdEncoder):
    """
    Concatenated Reed-Solomon Identification (RS2ID) encoder.
    
    RS2ID uses concatenated Reed-Solomon codes to provide enhanced error correction
    and identification capabilities. This system is particularly effective for
    applications requiring high reliability.
    
    Note: Current implementation supports single tag only (legacy version).
    
    Parameters:
        gf_exp: Base Galois field exponent (doubled due to concatenation)
        tag_pos: Outer code tag position (default: [2])
        tag_pos_in: Inner code tag position (default: [2])
    
    Warning:
        This is currently a legacy single-tag implementation. Multi-tag support
        will be added in future versions.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RS2ID encoder with default parameters."""
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2],
            "tag_pos_in": [2]
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize idcodes for concatenated Reed-Solomon operations."""
        # Effective GF exponent is doubled due to concatenation
        self.gf_exp = 2 * self.parameters["gf_exp"]
        self.idcodes = get_idcodes_instance(self.gf_exp)
        
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)
            self.idcodes.generate_gf_inner(self.gf_exp)
        elif self.gf_exp <= 32:
            self.idcodes.generate_gf_inner(self.gf_exp)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters and reinitialize idcodes."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> int:
        """
        Encode message using concatenated Reed-Solomon identification.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            Single identification tag (integer)
            
        Note:
            Currently returns single tag only. Multi-tag support planned.
        """
        if not message:
            raise ValueError("Message cannot be empty")
            
        tag_pos = self.parameters["tag_pos"][0]  # Use first position only
        tag_pos_in = self.parameters["tag_pos_in"][0]  # Use first inner position only
        
        result = self.idcodes.rs2id(message, tag_pos, tag_pos_in, self.gf_exp)
        return result


class RS2IDVerifier(IdVerifier):
    """
    Concatenated Reed-Solomon Identification (RS2ID) verifier.
    
    Verifies RS2ID tags by recomputing them using the internal encoder.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RS2ID verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = RS2IDEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: Message) -> bool:
        """
        Verify RS2ID codeword against a message.
        
        Args:
            codeword: Single identification tag to verify
            message: Message to check against
            
        Returns:
            True if codeword matches the recomputed tag
        """
        if not message:
            return False
            
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RMIDEncoder(IdEncoder):
    """
    Reed-Muller Identification (RMID) encoder.
    
    RMID is based on Reed-Muller codes, which are a class of linear error-correcting
    codes. They provide good performance for identification with configurable order
    parameters that affect the code's properties.
    
    Parameters:
        gf_exp: Galois field exponent (default: 8)
        tag_pos: List of tag positions for multi-tag systems (default: [2])
        rm_order: Reed-Muller code order (default: 1)
    
    Example:
        >>> encoder = RMIDEncoder({"gf_exp": 8, "tag_pos": [2], "rm_order": 2})
        >>> message = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> tags = encoder.encode(message)
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RMID encoder with default parameters."""
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2],
            "rm_order": 1
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize idcodes for Reed-Muller operations."""
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = get_idcodes_instance(self.gf_exp)
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)

    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters and reinitialize idcodes."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> List[int]:
        """
        Encode message using Reed-Muller identification.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            List of identification tags (one per tag position)
            
        Raises:
            ValueError: If message is empty or rm_order is invalid
        """
        if not message:
            raise ValueError("Message cannot be empty")
            
        rm_order = self.parameters["rm_order"]
        tags = []
        
        for tag_pos in self.parameters["tag_pos"]:
            tags.append(self.idcodes.rmid(message, tag_pos, rm_order, self.gf_exp))
        return tags


class RMIDVerifier(IdVerifier):
    """
    Reed-Muller Identification (RMID) verifier.
    
    Verifies RMID tags by recomputing them and comparing with provided codewords.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize RMID verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = RMIDEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codewords: List[int], message: Message) -> bool:
        """
        Verify RMID codewords against a message.
        
        Args:
            codewords: List of identification tags to verify
            message: Message to check against
            
        Returns:
            True if all codewords match the recomputed tags
        """
        if not message:
            return False
            
        recomputed_tags = self.encoder.encode(message)
        return all(tag == codewords[i] for i, tag in enumerate(recomputed_tags))


class SHA1IDEncoder(IdEncoder):
    """
    SHA1-based Identification encoder.
    
    This encoder uses the SHA1 cryptographic hash function to generate
    identification tags. It provides good distribution properties and is
    suitable for applications where cryptographic security is important.
    
    Parameters:
        gf_exp: Galois field exponent for output truncation (default: 8)
    
    Note:
        SHA1 is considered cryptographically weak for security applications
        but is still useful for identification purposes.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize SHA1ID encoder with default parameters."""
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize idcodes for SHA1 operations."""
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters and reinitialize idcodes."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> int:
        """
        Encode message using SHA1-based identification.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            Single identification tag derived from SHA1 hash
            
        Raises:
            ValueError: If message is empty
        """
        if not message:
            raise ValueError("Message cannot be empty")
            
        result = self.idcodes.sha1id(message, self.gf_exp)
        return result


class SHA1IDVerifier(IdVerifier):
    """
    SHA1-based Identification verifier.
    
    Verifies SHA1ID tags by recomputing the hash and comparing.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize SHA1ID verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = SHA1IDEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: Message) -> bool:
        """
        Verify SHA1ID codeword against a message.
        
        Args:
            codeword: Identification tag to verify
            message: Message to check against
            
        Returns:
            True if codeword matches the recomputed hash
        """
        if not message:
            return False
            
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA256IDEncoder(IdEncoder):
    """
    SHA256-based Identification encoder.
    
    This encoder uses the SHA256 cryptographic hash function to generate
    identification tags. SHA256 is more secure than SHA1 and provides
    excellent distribution properties.
    
    Parameters:
        gf_exp: Galois field exponent for output truncation (default: 8)
    
    Note:
        SHA256 is currently considered cryptographically secure and is
        recommended for applications requiring strong hash functions.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize SHA256ID encoder with default parameters."""
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize idcodes for SHA256 operations."""
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters and reinitialize idcodes."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> int:
        """
        Encode message using SHA256-based identification.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            Single identification tag derived from SHA256 hash
            
        Raises:
            ValueError: If message is empty
        """
        if not message:
            raise ValueError("Message cannot be empty")
            
        result = self.idcodes.sha256id(message, self.gf_exp)
        return result


class SHA256IDVerifier(IdVerifier):
    """
    SHA256-based Identification verifier.
    
    Verifies SHA256ID tags by recomputing the hash and comparing.
    """
    
    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize SHA256ID verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = SHA256IDEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: Message) -> bool:
        """
        Verify SHA256ID codeword against a message.
        
        Args:
            codeword: Identification tag to verify
            message: Message to check against
            
        Returns:
            True if codeword matches the recomputed hash
        """
        if not message:
            return False
            
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword
class NoCodeEncoder(IdEncoder):
    """
    No-code baseline encoder for comparison and testing.
    
    This encoder simply returns the first element of the message as the identification
    tag. It serves as a baseline for comparison with other identification systems
    and is useful for testing framework functionality.
    
    Parameters:
        gf_exp: Galois field exponent (not used, kept for interface compatibility)
    
    Note:
        This is a trivial encoder meant for testing and comparison purposes only.
        It provides no error correction or identification security.
    """

    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize NoCode encoder with default parameters."""
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self) -> None:
        """Initialize parameters (no idcodes library needed)."""
        self.gf_exp = self.parameters["gf_exp"]
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters."""
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Message) -> int:
        """
        Encode message by returning the first element.
        
        Args:
            message: Input message as list of integers
            
        Returns:
            First element of the message
            
        Raises:
            ValueError: If message is empty
        """
        if not message:
            raise ValueError("Message cannot be empty")
        return message[0]


class NoCodeVerifier(IdVerifier):
    """
    No-code baseline verifier for comparison and testing.
    
    This verifier checks if the codeword matches the first element of the message.
    It pairs with NoCodeEncoder for baseline testing.
    """

    def __init__(self, parameters: Optional[Parameters] = None) -> None:
        """Initialize NoCode verifier with matching encoder."""
        super().__init__(parameters)
        self.encoder = NoCodeEncoder(parameters)
    
    def set_parameters(self, parameters: Parameters) -> None:
        """Update parameters for both verifier and internal encoder."""
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: Message) -> bool:
        """
        Verify codeword by comparing with first message element.
        
        Args:
            codeword: Identification tag to verify
            message: Message to check against
            
        Returns:
            True if codeword equals the first message element
            
        Raises:
            ValueError: If message is empty
        """
        if not message:
            raise ValueError("Message cannot be empty")
        return codeword == message[0]


def create_id_system(
    system_type: str = "RSID", 
    parameters: Optional[Parameters] = None
) -> IdSystem:
    """
    Factory function to create identification systems.
    
    This function provides a convenient way to create identification systems
    without directly instantiating encoder and verifier classes.
    
    Args:
        system_type: Type of identification system to create.
                    Supported types: "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID", "NoCode"
        parameters: Dictionary of system-specific parameters
        
    Returns:
        Configured IdSystem instance
        
    Raises:
        ValueError: If system_type is not supported
        
    Example:
        >>> # Create RSID system with custom parameters
        >>> system = create_id_system("RSID", {
        ...     "gf_exp": 8,
        ...     "tag_pos": [2, 3]
        ... })
        >>> 
        >>> # Create SHA256 system with default parameters
        >>> hash_system = create_id_system("SHA256ID")
    """
    parameters = parameters or {}
    
    # Registry of available identification systems
    systems = {
        "RSID": (RSIDEncoder, RSIDVerifier),
        "RS2ID": (RS2IDEncoder, RS2IDVerifier),
        "RMID": (RMIDEncoder, RMIDVerifier),
        "SHA1ID": (SHA1IDEncoder, SHA1IDVerifier),
        "SHA256ID": (SHA256IDEncoder, SHA256IDVerifier),
        "NoCode": (NoCodeEncoder, NoCodeVerifier)
    }
    
    if system_type not in systems:
        raise ValueError(
            f"Unsupported system type: {system_type}. "
            f"Supported types: {list(systems.keys())}"
        )
    
    encoder_class, verifier_class = systems[system_type]
    encoder = encoder_class(parameters)
    verifier = verifier_class(parameters)
    
    return IdSystem(encoder, verifier)