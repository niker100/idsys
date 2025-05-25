#!/usr/bin/env python3
"""
Core module for identification systems using the idcodes library.

This module provides the fundamental classes and functions for
implementing and evaluating identification systems with multiple coding schemes.
"""
import numpy as np
import random
import string
from typing import List, Tuple, Any, Optional, Dict, Union
from idcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64


class IdEncoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def encode(self, message: Any) -> Any:
        raise NotImplementedError("Subclasses must implement encode method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdDecoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def decode(self, codeword: Any, message: Any) -> bool:
        raise NotImplementedError("Subclasses must implement decode method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdSystem:
    def __init__(self, encoder: IdEncoder, decoder: IdDecoder):
        self.encoder = encoder
        self.decoder = decoder

    def send(self, message: Any) -> Any:
        return self.encoder.encode(message)

    def receive(self, codeword: Any, message: Any) -> bool:
        return self.decoder.decode(codeword, message)


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


def _prepare_message_list(message: Any, alphabet_size: int = 256) -> List[int]:
    """Convert various message types to integer list for idcodes library."""
    if isinstance(message, list) and all(isinstance(x, int) for x in message):
        # Already a list of integers
        return [x % alphabet_size for x in message]
    elif isinstance(message, int):
        # Convert integer to bytes then to list
        if message == 0:
            return [0]
        byte_length = (message.bit_length() + 7) // 8
        byte_data = message.to_bytes(byte_length, 'big')
        return [b % alphabet_size for b in byte_data]
    elif isinstance(message, str):
        # Convert string to bytes then to list
        byte_data = message.encode('utf-8')
        return [b % alphabet_size for b in byte_data]
    elif isinstance(message, bytes):
        # Convert bytes to list
        return [b % alphabet_size for b in message]
    elif isinstance(message, np.ndarray):
        # Convert numpy array to list
        return [int(x) % alphabet_size for x in message.flatten()]
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


class RSIDEncoder(IdEncoder):
    """Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2,  # Changed from 0 to avoid collisions
            "alphabet_size": None  # Will be set based on gf_exp
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        if self.parameters["alphabet_size"] is None:
            self.parameters["alphabet_size"] = 2 ** self.gf_exp
        
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Any) -> int:
        msg_list = _prepare_message_list(message, self.parameters["alphabet_size"])
        tag_pos = self.parameters["tag_pos"]
        
        result = self.idcodes.rsid(msg_list, tag_pos, self.exp_arr, self.log_arr, self.gf_exp)
        return result


class RSIDDecoder(IdDecoder):
    """Reed-Solomon Identification decoder."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RSIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def decode(self, codeword: int, message: Any) -> bool:
        # Re-encode the message and compare
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RS2IDEncoder(IdEncoder):
    """Concatenated Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2,  # Standard position
            "tag_pos_in": 2,  # Standard position  
            "alphabet_size": None
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        if self.parameters["alphabet_size"] is None:
            self.parameters["alphabet_size"] = 2 ** self.gf_exp
            
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.idcodes.generate_gf_inner(self.gf_exp)
        
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
        self.exp_arr_in = self.idcodes.get_exp_arr_in()
        self.log_arr_in = self.idcodes.get_log_arr_in()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Any) -> int:
        msg_list = _prepare_message_list(message, self.parameters["alphabet_size"])
        tag_pos = self.parameters["tag_pos"]
        tag_pos_in = self.parameters["tag_pos_in"]
        
        result = self.idcodes.rs2id(
            msg_list, tag_pos, tag_pos_in, 
            self.exp_arr, self.log_arr, 
            self.exp_arr_in, self.log_arr_in, 
            self.gf_exp
        )
        return result


class RS2IDDecoder(IdDecoder):
    """Concatenated Reed-Solomon Identification decoder."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RS2IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def decode(self, codeword: int, message: Any) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RMIDEncoder(IdEncoder):
    """Reed-Muller Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2,  # Standard position
            "rm_order": 1,
            "alphabet_size": None
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        if self.parameters["alphabet_size"] is None:
            self.parameters["alphabet_size"] = 2 ** self.gf_exp
            
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Any) -> int:
        msg_list = _prepare_message_list(message, self.parameters["alphabet_size"])
        tag_pos = self.parameters["tag_pos"]
        rm_order = self.parameters["rm_order"]
        
        result = self.idcodes.rmid(msg_list, tag_pos, rm_order, self.exp_arr, self.log_arr, self.gf_exp)
        return result


class RMIDDecoder(IdDecoder):
    """Reed-Muller Identification decoder."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RMIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def decode(self, codeword: int, message: Any) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA1IDEncoder(IdEncoder):
    """SHA1-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "alphabet_size": None
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        if self.parameters["alphabet_size"] is None:
            self.parameters["alphabet_size"] = 2 ** self.gf_exp
            
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Any) -> int:
        msg_list = _prepare_message_list(message, self.parameters["alphabet_size"])
        result = self.idcodes.sha1id(msg_list, self.gf_exp)
        return result


class SHA1IDDecoder(IdDecoder):
    """SHA1-based Identification decoder."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA1IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def decode(self, codeword: int, message: Any) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA256IDEncoder(IdEncoder):
    """SHA256-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "alphabet_size": None
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        if self.parameters["alphabet_size"] is None:
            self.parameters["alphabet_size"] = 2 ** self.gf_exp
            
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: Any) -> int:
        msg_list = _prepare_message_list(message, self.parameters["alphabet_size"])
        result = self.idcodes.sha256id(msg_list, self.gf_exp)
        return result


class SHA256IDDecoder(IdDecoder):
    """SHA256-based Identification decoder."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA256IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def decode(self, codeword: int, message: Any) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


def create_id_system(system_type: str = "RSID", parameters: Optional[Dict[str, Any]] = None) -> IdSystem:
    """
    Create an identification system with the specified encoder type.
    
    Args:
        system_type: One of "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"
        parameters: System parameters
    """
    parameters = parameters or {}
    
    encoder_classes = {
        "RSID": (RSIDEncoder, RSIDDecoder),
        "RS2ID": (RS2IDEncoder, RS2IDDecoder),
        "RMID": (RMIDEncoder, RMIDDecoder),
        "SHA1ID": (SHA1IDEncoder, SHA1IDDecoder),
        "SHA256ID": (SHA256IDEncoder, SHA256IDDecoder)
    }
    
    if system_type not in encoder_classes:
        raise ValueError(f"Unsupported system type: {system_type}. "
                        f"Supported types: {list(encoder_classes.keys())}")
    
    encoder_class, decoder_class = encoder_classes[system_type]
    encoder = encoder_class(parameters)
    decoder = decoder_class(parameters)
    
    return IdSystem(encoder, decoder)


def generate_numeric_messages(count: int, min_value: int = 0, max_value: int = 1000000) -> List[int]:
    """Generate a list of random numeric messages."""
    return [random.randint(min_value, max_value) for _ in range(count)]


def generate_string_messages(count: int, length: int = 10, alphabet_size: int = 62) -> List[str]:
    """Generate a list of random string messages with specified alphabet size."""
    alphabets = {
        2: "01",
        3: "ABC", 
        4: "ACTG",
        8: "01234567",
        16: "0123456789ABCDEF",
        26: string.ascii_lowercase,
        52: string.ascii_letters,
        62: string.ascii_letters + string.digits,
        95: string.printable.strip()
    }
    
    if alphabet_size in alphabets:
        chars = alphabets[alphabet_size]
    else:
        # Generate alphabet of requested size
        chars = (string.ascii_letters + string.digits + string.punctuation)[:alphabet_size]
        if len(chars) < alphabet_size:
            # Repeat characters if needed
            chars = chars * ((alphabet_size // len(chars)) + 1)
            chars = chars[:alphabet_size]
    
    return [''.join(random.choice(chars) for _ in range(length)) for _ in range(count)]


def generate_test_messages(count: int, length: int, alphabet_size: int = 62, 
                          message_type: str = "string", seed: Optional[int] = None) -> List[Any]:
    """
    Generate test messages for evaluation.
    
    Args:
        count: Number of messages to generate
        length: Length of each message (in characters for strings, value range for integers)
        alphabet_size: Size of alphabet for string messages
        message_type: "string", "numeric", or "bytes"
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    if message_type == "string":
        return generate_string_messages(count, length, alphabet_size)
    elif message_type == "numeric":
        max_val = min(alphabet_size ** length - 1, 2**32 - 1)  # Prevent overflow
        return generate_numeric_messages(count, 0, max_val)
    elif message_type == "bytes":
        return [bytes([random.randint(0, alphabet_size-1) for _ in range(length)]) 
                for _ in range(count)]
    else:
        raise ValueError(f"Unsupported message type: {message_type}")
