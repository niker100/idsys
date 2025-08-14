"""
IDSYS - Identification Systems Analysis Framework

Copyright (c) 2025 niker100
Licensed under the MIT License - see LICENSE file for details
"""

__version__ = "0.1.0"

# Import key components to the top level
from .core.idsystems import (
    create_id_system,
    IdSystem
)
from .core.message_generation import (
    generate_test_messages,
    generate_structured_messages
)
from .core.metrics import IdMetrics
from .utils.checkpoint import create_checkpoint_manager

__all__ = [
    "create_id_system",
    "generate_test_messages",
    "generate_structured_messages",
    "IdMetrics",
    "create_checkpoint_manager",
    "IdSystem"
]