"""
Demo 01: Understanding Attention - Simplified

This demo introduces attention in 15 minutes:
1. What is attention? (analogy)
2. Quick visual example
3. Use Keras MultiHeadAttention

NO deep math. Focus on INTUITION.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ATTENTION: The Key Idea")
print("=" * 60)

print("""
SIMPLE ANALOGY: Reading Comprehension

Question: "What did the CAT do?"
Sentence: "The cat sat on the mat."

Your brain FOCUSES on "cat" and "sat" - not every word equally.
That's attention!

In neural networks:
- Attention lets the model FOCUS on relevant parts
- Each position can "look at" every other position
- Weights show what the model pays attention to
""")

# ============================================================================
# PART 1: Visual Example
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: Attention Weights - What Does the Model Focus On?")
print("=" * 60)

# Simple attention pattern for "The cat sat on the mat"
words = ["The", "cat", "sat", "on", "the", "mat"]

# When processing "sat", what does it attend to?
print("""
When the model processes the word "sat", it asks:
  "Which other words should I pay attention to?"

Attention weights might look like:
  "The"  -> 5%   (not very relevant)
  "cat"  -> 40%  (who sat? important!)
  "sat"  -> 25%  (myself)
  "on"   -> 5%   (just a preposition)
  "the"  -> 5%   (not relevant)
  "mat"  -> 20%  (where? somewhat important)
""")

# Visualize
attention_for_sat = [0.05, 0.40, 0.25, 0.05, 0.05, 0.20]

plt.figure(figsize=(10, 3))
plt.bar(words, attention_for_sat, color='steelblue')
plt.title('What "sat" attends to', fontsize=14)
plt.ylabel('Attention Weight')
plt.ylim(0, 0.5)
for i, v in enumerate(attention_for_sat):
    plt.text(i, v + 0.02, f'{v:.0%}', ha='center')
plt.tight_layout()
plt.savefig('attention_example.png', dpi=150)
print("[OK] Saved attention_example.png")
plt.close()

# ============================================================================
# PART 2: Using Keras Attention
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: Using Attention in Keras (2 lines!)")
print("=" * 60)

print("""
Good news: Keras makes attention EASY!

Just use: layers.MultiHeadAttention(num_heads=4, key_dim=32)
""")

# Simple example
seq_length = 10
d_model = 64

# Create attention layer
attention = layers.MultiHeadAttention(
    num_heads=4,      # 4 different "perspectives"
    key_dim=16,       # Size of each perspective
    dropout=0.1       # Regularization
)

# Test it
test_input = tf.random.normal((2, seq_length, d_model))
output = attention(test_input, test_input)

print(f"Input shape:  {test_input.shape} (batch, sequence, features)")
print(f"Output shape: {output.shape} (same shape!)")
print("\nThat's it! Attention transforms your sequence.")

# ============================================================================
# PART 3: Why Attention Beats RNNs
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: Why Attention > RNNs")
print("=" * 60)

print("""
RNN Problem:
  Word 1 -> Word 2 -> Word 3 -> ... -> Word 100
  By Word 100, information about Word 1 is mostly LOST!

Attention Solution:
  Every word can directly connect to every other word.
  Word 100 can "see" Word 1 with just ONE step!

Benefits:
  [+] No information loss over distance
  [+] Parallelizable (faster training)
  [+] Interpretable (attention weights show what model focuses on)
""")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. Attention = letting the model FOCUS on relevant parts
2. Each position can attend to ALL other positions
3. Attention weights show what the model "looks at"
4. In Keras: layers.MultiHeadAttention()
5. This is the foundation of ChatGPT, BERT, etc.

Next demo: Building a simple Transformer!
""")
