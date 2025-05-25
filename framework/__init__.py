#!/usr/bin/env python3
"""
Identification System Framework Package.

This package provides tools for creating, evaluating, and visualizing 
identification systems based on various coding schemes.
"""

from .core import (
    IdEncoder, IdDecoder, IdSystem,
    RSIDEncoder, RSIDDecoder,
    RS2IDEncoder, RS2IDDecoder, 
    RMIDEncoder, RMIDDecoder,
    SHA1IDEncoder, SHA1IDDecoder,
    SHA256IDEncoder, SHA256IDDecoder,
    create_id_system, 
    generate_test_messages,
    _get_idcodes_instance
)

from .metrics import IdMetrics

from .utils import (
    evaluate_system_with_generated_messages,
    batch_evaluate_parameters
)

__all__ = [
    # Core classes
    'IdEncoder', 'IdDecoder', 'IdSystem',
    
    # Encoder/Decoder implementations
    'RSIDEncoder', 'RSIDDecoder',
    'RS2IDEncoder', 'RS2IDDecoder', 
    'RMIDEncoder', 'RMIDDecoder',
    'SHA1IDEncoder', 'SHA1IDDecoder',
    'SHA256IDEncoder', 'SHA256IDDecoder',
    
    # Factory and utility functions
    'create_id_system',
    'generate_string_sequence',
    'generate_test_messages',
    'read_file_data',
    'read_text_sequence',
    '_get_idcodes_instance',
    
    # Metrics
    'IdMetrics',
    
    # Utilities
    'evaluate_system_with_generated_messages',
    'batch_evaluate_parameters'
]