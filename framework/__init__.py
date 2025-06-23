#!/usr/bin/env python3
"""
Identification System Framework Package.

This package provides tools for creating, evaluating, and visualizing 
identification systems based on various coding schemes.
"""

from .core import (
    IdEncoder, IdVerifier, IdSystem,
    RSIDEncoder, RSIDVerifier,
    RS2IDEncoder, RS2IDVerifier, 
    RMIDEncoder, RMIDVerifier,
    SHA1IDEncoder, SHA1IDVerifier,
    SHA256IDEncoder, SHA256IDVerifier,
    create_id_system, 
    generate_test_messages,
    _get_idcodes_instance
)

from .metrics import IdMetrics

from .utils import (
    batch_evaluate_parameters
)

from .checkpoint import (
    AnalysisCheckpoint,
    create_checkpoint_manager
)

__all__ = [
    # Core classes
    'IdEncoder', 'IdVerifier', 'IdSystem',
    
    # Encoder/Verifier implementations
    'RSIDEncoder', 'RSIDVerifier',
    'RS2IDEncoder', 'RS2IDVerifier', 
    'RMIDEncoder', 'RMIDVerifier',
    'SHA1IDEncoder', 'SHA1IDVerifier',
    'SHA256IDEncoder', 'SHA256IDVerifier',
    
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
    'batch_evaluate_parameters',
    
    # Checkpointing
    'AnalysisCheckpoint',
    'create_checkpoint_manager',
    
    # Checkpointing
    'AnalysisCheckpoint',
    'create_checkpoint_manager'
]