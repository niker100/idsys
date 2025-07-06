#!/usr/bin/env python3
"""
Core module for identification systems using the idcodes library.

This module provides the fundamental classes and functions for
implementing and evaluating identification systems with multiple coding schemes.
"""
from typing import List, Any, Optional, Dict, Set, Tuple
from ecidcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64
import numpy as np


class IdEncoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def encode(self, message: Any) -> Any:
        raise NotImplementedError("Subclasses must implement encode method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdVerifier:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def verify(self, codeword: Any, message: Any) -> bool:
        raise NotImplementedError("Subclasses must implement verify method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdSystem:
    def __init__(self, encoder: IdEncoder, verifier: IdVerifier):
        self.encoder = encoder
        self.verifier = verifier

    def send(self, message: Any) -> Any:
        return self.encoder.encode(message)

    def receive(self, codeword: Any, message: Any) -> bool:
        return self.verifier.verify(codeword, message)
    
    def receive_k(self, codeword: Any, messages: List[Any]) -> bool:
        """
        Verify multiple messages against a single codeword.
        
        Args:
            codeword: The codeword to verify against
            messages: List of messages to verify
            
        Returns:
            True if at least one message verifies against the codeword, False otherwise
        """
        for msg in messages:
            if self.verifier.verify(codeword, msg):
                return msg
        return None


def _get_idcodes_instance(gf_exp: int):
    """Get the appropriate IDCODES instance based on Galois field exponent."""
    if gf_exp <= 8:
        return IDCODES_U8()
    elif gf_exp <= 16:
        return IDCODES_U16()
    elif gf_exp <= 32:
        return IDCODES_U32()
    elif gf_exp <= 64:
        return IDCODES_U64()
    else:
        raise ValueError(f"Unsupported GF exponent: {gf_exp}. Must be <= 64")


class RSIDEncoder(IdEncoder):
    """Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2]
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)
            self.idcodes.initialize_gf(self.idcodes.get_exp_arr(), self.idcodes.get_log_arr(), self.gf_exp)            
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> List[int]:
        tags = []
        for tag_pos in self.parameters["tag_pos"]:
            tags.append(self.idcodes.rsid(message, tag_pos, self.gf_exp))
        return tags


class RSIDVerifier(IdVerifier):
    """Reed-Solomon Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RSIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codewords: List[int], message: List[int]) -> bool:
        recomputed_tags = self.encoder.encode(message)
        return all(tag == codewords[i] for i, tag in enumerate(recomputed_tags))

# WARNING: Still old version with only one tag
class RS2IDEncoder(IdEncoder):
    """Concatenated Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2],
            "tag_pos_in": [2]
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = 2 * self.parameters["gf_exp"] # Due to concatenation
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)
            self.idcodes.generate_gf_inner(self.gf_exp)
            self.idcodes.initialize_gf(self.idcodes.get_exp_arr(), self.idcodes.get_log_arr(), self.gf_exp)
        elif self.gf_exp <= 32:
            self.idcodes.generate_gf_inner(self.gf_exp)
            # self.idcodes.initialize_gf(self.idcodes.get_exp_arr_in(), self.idcodes.get_log_arr_in(), self.gf_exp // 2)
    
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        tag_pos = self.parameters["tag_pos"]
        tag_pos_in = self.parameters["tag_pos_in"]
        
        result = self.idcodes.rs2id(message, tag_pos[0], tag_pos_in[0], self.gf_exp)
        return result


class RS2IDVerifier(IdVerifier):
    """Concatenated Reed-Solomon Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RS2IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RMIDEncoder(IdEncoder):
    """Reed-Muller Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": [2],
            "rm_order": 1
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        if self.gf_exp <= 16:
            self.idcodes.generate_gf_outer(self.gf_exp)
            self.idcodes.initialize_gf(self.idcodes.get_exp_arr(), self.idcodes.get_log_arr(), self.gf_exp)


    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> List[int]:
        rm_order = self.parameters["rm_order"]
        tags = []
        
        for tag_pos in self.parameters["tag_pos"]:
            # print(f"Encoding {message} RMID with tag_pos={tag_pos}, rm_order={rm_order}, gf_exp={self.gf_exp}")
            tags.append(self.idcodes.rmid(message, tag_pos, rm_order, self.gf_exp))
        return tags


class RMIDVerifier(IdVerifier):
    """Reed-Muller Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RMIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codewords: List[int], message: List[int]) -> bool:
        recomputed_tags = self.encoder.encode(message)
        return all(tag == codewords[i] for i, tag in enumerate(recomputed_tags))


class SHA1IDEncoder(IdEncoder):
    """SHA1-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        result = self.idcodes.sha1id(message, self.gf_exp)
        return result


class SHA1IDVerifier(IdVerifier):
    """SHA1-based Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA1IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA256IDEncoder(IdEncoder):
    """SHA256-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        result = self.idcodes.sha256id(message, self.gf_exp)
        return result


class SHA256IDVerifier(IdVerifier):
    """SHA256-based Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA256IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword
    
class NoCodeEncoder(IdEncoder):
    """No-code encoder that simply returns the first element of the message."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        if not message:
            raise ValueError("Message cannot be empty")
        return message[0]

class NoCodeVerifier(IdVerifier):
    """No-code verifier that checks if the codeword matches the first element of the message."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = NoCodeEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        if not message:
            raise ValueError("Message cannot be empty")
        return codeword == message[0]

def create_id_system(system_type: str = "RSID", parameters: Optional[Dict[str, Any]] = None) -> IdSystem:
    """
    Create an identification system with the specified encoder/verifier type.
    
    Args:
        system_type: One of "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"
        parameters: System parameters (gf_exp, tag_pos, tag_pos_in, rm_order)
    """
    parameters = parameters or {}
    
    systems = {
        "RSID": (RSIDEncoder, RSIDVerifier),
        "RS2ID": (RS2IDEncoder, RS2IDVerifier),
        "RMID": (RMIDEncoder, RMIDVerifier),
        "SHA1ID": (SHA1IDEncoder, SHA1IDVerifier),
        "SHA256ID": (SHA256IDEncoder, SHA256IDVerifier),
        "NoCode": (NoCodeEncoder, NoCodeVerifier)
    }
    
    if system_type not in systems:
        raise ValueError(f"Unsupported system type: {system_type}. "
                        f"Supported types: {list(systems.keys())}")
    
    encoder_class, verifier_class = systems[system_type]
    encoder = encoder_class(parameters)
    verifier = verifier_class(parameters)
    
    return IdSystem(encoder, verifier)


def generate_test_messages(vec_len: int, gf_exp: int, count: int = 1) -> List[List[int]]:
    """
    Generate test messages using the same approach as Pybenchmark.
    
    Args:
        vec_len: Vector length in elements
        gf_exp: Galois field exponent
        count: Number of messages to generate
        
    Returns:
        List of integer lists representing messages
    """
    # Adjust vector length based on GF exponent (like Pybenchmark)
    if gf_exp >= 33:
        vec_len_ = vec_len // 8
    elif gf_exp >= 17:
        vec_len_ = vec_len // 4
    elif gf_exp >= 9:
        vec_len_ = vec_len // 2
    else:
        vec_len_ = vec_len
    
    # Get appropriate idcodes instance
    Id = _get_idcodes_instance(gf_exp)
    
    messages = []
    for _ in range(count):
        message = Id.generate_string_sequence(vec_len_)
        messages.append(message)
    
    return messages

def generate_structured_messages(
    vec_len: int,
    pattern_type: str,
    gf_exp: int,
    target_count: int = 5000,
    generate_first: bool = False,
    seed: Optional[int] = None,
    worker_offset: int = 0
):
    """
    Generator factory that produces messages with specified structural patterns.
    
    Args:
        vec_len: Vector length in elements
        pattern_type: Type of pattern to generate ('random', 'incremental', etc.)
        gf_exp: Galois field exponent
        target_count: Number of messages to generate
        generate_first: Whether to yield the first message
        seed: Random seed for reproducibility
        worker_offset: Offset for multiprocessing (use worker's ID/index)
    """
    # Static variable to store the first message
    if not hasattr(generate_structured_messages, "_first_message"):
        generate_structured_messages._first_message = None
    
    # Use worker_offset to create process-specific randomness
    # but keep global seed for reproducibility if provided
    if seed is not None:
        base_seed = seed
    else:
        # Use current time if no seed provided
        import time
        base_seed = int(time.time())
    
    # Create a process-specific random state
    process_seed = base_seed + worker_offset
    process_random = np.random.RandomState(process_seed)

    Id = _get_idcodes_instance(gf_exp)
    if gf_exp >= 33:
        vec_len_ = vec_len // 8
    elif gf_exp >= 17:
        vec_len_ = vec_len // 4
    elif gf_exp >= 9:
        vec_len_ = vec_len // 2
    else:
        vec_len_ = vec_len

    # Helper to generate a single message for a given attempt
    def _gen_pattern(attempt):
        # Use process-specific attempt counter
        effective_attempt = attempt + (worker_offset * 1000)  # Large offset to avoid overlaps
        
        if pattern_type == 'random':
            
            msg = Id.generate_string_sequence(vec_len_)           

            return msg
            
        elif pattern_type == "incremental":
            return [0] * (vec_len - 1) + [effective_attempt % (2 ** gf_exp)]
            
        elif pattern_type == "repeated_patterns":
            # Select a base pattern based on the worker's offset to ensure
            # different workers generate fundamentally different sequences.
            base_patterns = [
                [255, 0],
                [170, 187],          # 0xAA, 0xBB
                [85, 170],           # 0x55, 0xAA
                [1, 2, 3, 4],
                [15, 240, 15, 240],  # 0x0F, 0xF0
                [10, 20, 30, 40]
            ]
            base_pattern = base_patterns[worker_offset % len(base_patterns)]
            
            # Modify the base pattern with the attempt number to generate unique messages.
            # This ensures that each message within a worker's sequence is unique.
            pattern = [(p + effective_attempt) % (2**gf_exp) for p in base_pattern]
            
            # Tile the generated pattern to fill the message vector.
            num_repeats = (vec_len + len(pattern) - 1) // len(pattern)
            msg = (pattern * num_repeats)[:vec_len]
            return msg
            
        elif pattern_type == "sparse":
            msg = [0] * vec_len
            num_nonzero = 1 + (effective_attempt % 3)
            # Use different prime numbers for each worker
            prime_offset = [7, 11, 13, 17, 19, 23, 29, 31][worker_offset % 8]
            positions = [(effective_attempt + j * prime_offset) % vec_len for j in range(num_nonzero)]
            for pos in positions:
                msg[pos] = 1 + (effective_attempt + pos) % (2 ** gf_exp - 1)
            return msg
            
        elif pattern_type == "low_entropy":
            # Use process-specific random state
            alphabet = [0, 1, 2, 3]
            return process_random.choice(alphabet, size=vec_len).tolist()
        
        elif pattern_type == "only_two":
            # Generate messages with only two distinct values
            if effective_attempt % 2 == 0:
                return [0] * vec_len
            else:
                return [1] * vec_len
            
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

    # Generate and store/yield the first message if requested
    if generate_first or generate_structured_messages._first_message is None:
        # First message should be the same for all processes
        first_message = _gen_pattern(0) if worker_offset == 0 else generate_structured_messages._first_message
        generate_structured_messages._first_message = first_message
        if generate_first:
            yield first_message
            return  # Only yield the first message if requested

    first_message = generate_structured_messages._first_message

    # Now generate messages, skipping any that match the first message
    count = 0
    attempts = 1  # Start from 1 to avoid duplicating the first message
    max_attempts = target_count * 10  # Avoid infinite loops

    while count < target_count and attempts < max_attempts:
        msg = _gen_pattern(attempts)
        attempts += 1
        if msg != first_message:
            yield msg
            count += 1  # Fix: increment the counter