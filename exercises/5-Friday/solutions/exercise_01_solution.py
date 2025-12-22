"""Solution for Exercise 01"""
import tensorflow as tf
from tensorflow import keras
from keras import layers

sequence_length = 10
d_model = 64
batch_size = 2
sample_input = tf.random.normal((batch_size, sequence_length, d_model))

# SOLUTION
attention_layer = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)
output = attention_layer(sample_input, sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
print("[OK] Complete!")
