"""
Exercise 02: Build a Transformer

Complete the TODO sections to build a text classifier.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

print("=" * 50)
print("Exercise 02: Build a Transformer")
print("=" * 50)

# Configuration
vocab_size = 10000
maxlen = 50
num_classes = 2

def build_transformer():
    """Build a simple Transformer classifier."""
    
    inputs = keras.Input(shape=(maxlen,))
    
    # ========================================================================
    # TODO 1: Add embedding layer
    #
    # Hint: layers.Embedding(vocab_size, 64)
    # ========================================================================
    
    x = None  # TODO: Add embedding
    
    # ========================================================================
    # TODO 2: Add position embeddings
    #
    # Hint: 
    #   positions = tf.range(maxlen)
    #   pos_embed = layers.Embedding(maxlen, 64)(positions)
    #   x = x + pos_embed
    # ========================================================================
    
    # TODO: Add position embeddings here
    
    # ========================================================================
    # TODO 3: Add MultiHeadAttention
    #
    # Hint: 
    #   attention_output = layers.MultiHeadAttention(
    #       num_heads=4, key_dim=16
    #   )(x, x)
    # ========================================================================
    
    # TODO: Add attention here
    
    # ========================================================================
    # TODO 4: Add classification head
    #
    # Hint:
    #   x = layers.GlobalAveragePooling1D()(x)
    #   outputs = layers.Dense(num_classes, activation='softmax')(x)
    # ========================================================================
    
    outputs = None  # TODO: Add classification head
    
    return keras.Model(inputs, outputs)

# Build and test
try:
    model = build_transformer()
    model.summary()
    
    # Test prediction
    test_input = np.random.randint(1, vocab_size, size=(1, maxlen))
    prediction = model.predict(test_input, verbose=0)
    print(f"\nTest prediction: {prediction}")
    print("\n✓ Exercise 02 complete!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("Complete the TODOs above")
