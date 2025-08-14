#!/usr/bin/env python3
# filepath: /workspaces/idsys/examples/basic_usage.py
"""
Basic usage example for the IDSYS framework.

This example demonstrates creating a system, generating test messages,
and performing basic identification operations.
"""

import numpy as np
from idsys import create_id_system, generate_test_messages

def main():
    print("== IDSYS Basic Usage Example ==")
    
    # Create an identification system
    # RSID = Reed-Solomon Identification system with:
    # - 8-bit Galois field (2^8 = 256 values)
    # - Tag at position 2
    system = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})
    
    # Generate test messages - 16 bytes each
    messages = generate_test_messages(vec_len=16, gf_exp=8, count=5)
    print(f"Generated {len(messages)} test messages")
    
    # Show the first message
    print(f"First message (truncated): {messages[0][:8]}...")
    
    # Encode a message to get an identification tag
    message = messages[0]
    tag = system.send(message)
    print(f"Generated tag: {tag}")
    
    # Verify the tag against the original message (should be True)
    is_valid = system.receive(tag, message)
    print(f"Tag is valid for original message: {is_valid}")
    
    # Verify against a different message (should be False)
    is_valid = system.receive(tag, messages[1])
    print(f"Tag is valid for different message: {is_valid}")

if __name__ == "__main__":
    main()