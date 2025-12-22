"""
Exercise 01: Attention Basics

Complete the TODO sections to use MultiHeadAttention.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

print("=" * 50)
print("Exercise 01: Attention Basics")
print("=" * 50)

# Sample input: batch of 2 sequences, 10 positions, 64 features
sequence_length = 10
d_model = 64
batch_size = 2

sample_input = tf.random.normal((batch_size, sequence_length, d_model))
print(f"Input shape: {sample_input.shape}")

# ============================================================================
# TODO 1: Create a MultiHeadAttention layer
# 
# Parameters:
#   - num_heads: 4
#   - key_dim: 16
#   - dropout: 0.1
# ============================================================================

attention_layer = None  # TODO: Replace with MultiHeadAttention(...)

# ============================================================================
# TODO 2: Apply self-attention
#
# For self-attention, both query and value are the same input
# ============================================================================

output = None  # TODO: Apply attention to sample_input

# ============================================================================
# Verification
# ============================================================================

if output is not None:
    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {output.shape == sample_input.shape}")
    print("\n✓ Exercise 01 complete!")
else:
    print("\n✗ Complete the TODOs above")
