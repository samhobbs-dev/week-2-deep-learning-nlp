# Transformer Architecture

## Learning Objectives
- Understand the Transformer architecture and its key components
- Learn how multi-head attention works and why it's powerful
- Understand the importance of positional encoding for sequence order
- Build Transformer encoder blocks using Keras

## Why This Matters

The Transformer architecture has become the foundation of modern AI. Understanding it is essential because:

- **Powering LLMs**: GPT-4, Claude, Gemini, and other LLMs are all based on Transformers
- **State-of-the-art NLP**: BERT, RoBERTa, T5 - all Transformers
- **Beyond text**: Vision Transformers (ViT), audio, protein folding (AlphaFold)
- **Industry standard**: Every major AI company uses Transformer-based models
- **Career essential**: Understanding Transformers is now a core ML skill

Building on yesterday's RNN/LSTM knowledge and today's attention mechanism, the Transformer represents the current peak of sequence modeling - achieving better results while being more parallelizable.

## The Transformer: "Attention Is All You Need"

### Removing Recurrence Entirely

The Transformer (Vaswani et al., 2017) made a bold claim: you don't need RNNs at all. Attention alone is sufficient.

```
RNN approach:
  Word 1 → Word 2 → Word 3 → Word 4 → Word 5
  (Sequential: must process in order)

Transformer approach:
  Word 1   Word 2   Word 3   Word 4   Word 5
     ↓       ↓        ↓        ↓        ↓
  [============ Self-Attention ============]
     ↓       ↓        ↓        ↓        ↓
  (Parallel: all positions processed simultaneously)
```

### Architecture Overview

```
                        Outputs
                           ↑
                    ┌──────────────┐
                    │   Decoder    │ × N
                    │   Block      │
                    └──────────────┘
                           ↑
    Encoder          ┌─────────────┐
    Output ─────────→│Cross-Attn   │
                     └─────────────┘
        
                    ┌──────────────┐
                    │   Encoder    │ × N
                    │   Block      │
                    └──────────────┘
                           ↑
                   Positional Encoding
                           ↑
                    Input Embedding
                           ↑
                        Inputs
```

## Multi-Head Attention

### Why Multiple Heads?

Single attention can only capture one type of relationship at a time. Multi-head attention runs **multiple attention operations in parallel**, each learning different patterns.

```
Single attention might capture:
  "cat" → "sat" (subject-verb)

Multi-head attention captures simultaneously:
  Head 1: Subject-verb relationships ("cat" → "sat")
  Head 2: Noun-determiner relationships ("cat" → "the")
  Head 3: Object relationships ("sat" → "mat")
  Head 4: Positional patterns
  ...
```

### How Multi-Head Works

```
Input: (batch, seq_len, d_model)  e.g., (32, 100, 512)

1. Split into h heads:
   d_k = d_model / h = 512 / 8 = 64  (per-head dimension)
   
2. For each head i:
   Q_i = input × W_Q_i   (project to 64-dim)
   K_i = input × W_K_i
   V_i = input × W_V_i
   head_i = Attention(Q_i, K_i, V_i)

3. Concatenate all heads:
   MultiHead = Concat(head_1, ..., head_8)  # Back to 512-dim
   
4. Final projection:
   Output = MultiHead × W_O
```

### Multi-Head Attention in Keras

```python
from tensorflow import keras
from keras import layers

# Keras built-in MultiHeadAttention
seq_length = 50
d_model = 512
num_heads = 8

# Input shape: (batch, seq_length, d_model)
inputs = keras.Input(shape=(seq_length, d_model))

# Multi-Head Self-Attention
attention_output = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=d_model // num_heads,  # d_k per head
    dropout=0.1
)(inputs, inputs)  # Self-attention: query=inputs, value=inputs

print(f"Input shape: {inputs.shape}")        # (None, 50, 512)
print(f"Output shape: {attention_output.shape}")  # (None, 50, 512)
```

## Positional Encoding

### The Problem: No Order Information

Unlike RNNs, Transformers process all positions in parallel. Without explicit position information:

```
"The cat sat on the mat"
"mat the on sat cat The"

To pure attention, these look identical!
(Same words, just different positions)
```

### The Solution: Add Position Information

We add a **positional encoding** to each embedding, giving the model position awareness.

```
Word embedding:     [0.5, -0.3, 0.8, ...]    (semantic info)
Position encoding:  [0.0, 1.0, 0.0, ...]    (position info)
─────────────────────────────────────────
Input to model:     [0.5, 0.7, 0.8, ...]    (combined)
```

### Sinusoidal Positional Encoding

The original Transformer used sine/cosine functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: Position in sequence (0, 1, 2, ...)
- i: Dimension index
- d_model: Embedding dimension
```

**Why sinusoids?**
1. Each position gets a unique encoding
2. Relative positions can be learned (PE[pos+k] is a linear function of PE[pos])
3. Can extrapolate to longer sequences than seen during training

### Implementing Positional Encoding

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

def get_positional_encoding(max_seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    positions = np.arange(max_seq_len)[:, np.newaxis]  # (seq_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]            # (1, d_model)
    
    # Compute angles
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    
    # Apply sin to even indices, cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])  # sin for even
    angles[:, 1::2] = np.cos(angles[:, 1::2])  # cos for odd
    
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, seq, d_model)

# Example: 100 positions, 512 dimensions
pos_encoding = get_positional_encoding(100, 512)
print(f"Positional encoding shape: {pos_encoding.shape}")  # (1, 100, 512)

# Usage in model
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, max_seq_len, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Scale embedding and add positional encoding
        embedded = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return embedded + self.pos_encoding[:, :seq_len, :]
```

## Transformer Encoder Block

### Components

Each encoder block has two sub-layers:

```
    Input
      ↓
┌─────────────────────┐
│  Multi-Head         │
│  Self-Attention     │
└─────────────────────┘
      ↓
   Add & Norm (residual connection + layer normalization)
      ↓
┌─────────────────────┐
│  Feed-Forward       │
│  Network            │
└─────────────────────┘
      ↓
   Add & Norm
      ↓
    Output
```

### Complete Encoder Block Implementation

```python
from tensorflow import keras
from keras import layers

class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),  # Expand
            layers.Dense(d_model),                     # Project back
            layers.Dropout(dropout_rate)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)  # Residual + normalize
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)   # Residual + normalize
        
        return x

# Example usage
d_model = 512
num_heads = 8
ff_dim = 2048  # Typically 4x d_model
seq_len = 50

encoder_block = TransformerEncoderBlock(d_model, num_heads, ff_dim)

# Test with random input
test_input = tf.random.normal((2, seq_len, d_model))  # (batch, seq, d_model)
output = encoder_block(test_input)
print(f"Input shape: {test_input.shape}")   # (2, 50, 512)
print(f"Output shape: {output.shape}")      # (2, 50, 512)
```

### Feed-Forward Network

The FFN expands and contracts the representation:

```
Input: 512-dim
  → Dense(2048, relu)   # Expand to 2048
  → Dense(512)          # Contract back to 512
Output: 512-dim

Why expand?
- Provides more capacity for non-linear transformations
- Acts like a 1x1 convolution over positions
- Each position processed independently (unlike attention)
```

### Layer Normalization vs Batch Normalization

```
Batch Norm: Normalize across batch dimension
  - Problematic for variable-length sequences
  - Depends on batch statistics

Layer Norm: Normalize across feature dimension
  - Works per-example, independent of batch
  - Standard in Transformers
```

```python
# Layer normalization
x = layers.LayerNormalization(epsilon=1e-6)(x)

# Normalizes across the last dimension (features)
# For shape (batch, seq, d_model), normalizes over d_model
```

## Stacking Encoder Blocks

```python
def build_transformer_encoder(
    vocab_size,
    seq_length,
    d_model=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=6,
    dropout_rate=0.1
):
    # Input layer
    inputs = keras.Input(shape=(seq_length,))
    
    # Embedding + positional encoding
    x = layers.Embedding(vocab_size, d_model)(inputs)
    x = x * tf.math.sqrt(tf.cast(d_model, tf.float32))  # Scale
    
    # Add positional encoding
    positions = tf.range(seq_length)
    pos_embedding = layers.Embedding(seq_length, d_model)(positions)
    x = x + pos_embedding
    
    x = layers.Dropout(dropout_rate)(x)
    
    # Stack encoder blocks
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)(x)
    
    return keras.Model(inputs, x)

# Create 6-layer transformer encoder
encoder = build_transformer_encoder(
    vocab_size=10000,
    seq_length=100,
    num_layers=6
)
encoder.summary()
```

## Transformer Decoder (Brief Overview)

The decoder has similar structure but with additional components:

```
                Input (shifted right)
                        ↓
                Positional Embedding
                        ↓
┌─────────────────────────────────────────┐
│  Masked Multi-Head Self-Attention       │  ← Can only attend to previous positions
│  (prevents looking at future tokens)    │
└─────────────────────────────────────────┘
                        ↓
                   Add & Norm
                        ↓
┌─────────────────────────────────────────┐
│  Cross-Attention                        │  ← Attends to encoder output
│  (query from decoder, key/value from    │
│   encoder)                              │
└─────────────────────────────────────────┘
                        ↓
                   Add & Norm
                        ↓
┌─────────────────────────────────────────┐
│  Feed-Forward Network                   │
└─────────────────────────────────────────┘
                        ↓
                   Add & Norm
                        ↓
                     Output
```

**Note**: For classification tasks (like sentiment analysis), we typically only need the encoder!

## Key Takeaways

1. **Transformers use attention only** - no recurrence needed
2. **Multi-head attention** captures multiple relationship types in parallel
3. **Positional encoding** provides sequence order information since attention itself is position-agnostic
4. **Encoder blocks** consist of: Multi-Head Attention → Add & Norm → FFN → Add & Norm
5. **Residual connections** help gradient flow; Layer Norm stabilizes training
6. **Feed-forward networks** provide non-linear transformations at each position
7. **Parallelization** - Transformers are much faster to train than RNNs
8. **Scaled dot-product** attention prevents gradient problems with large dimensions

## Common Transformer Configurations

| Model | Layers | d_model | Heads | FFN dim | Parameters |
|-------|--------|---------|-------|---------|------------|
| BERT-base | 12 | 768 | 12 | 3072 | 110M |
| BERT-large | 24 | 1024 | 16 | 4096 | 340M |
| GPT-2 | 12-48 | 768-1600 | 12-25 | 3072-6400 | 117M-1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |

## External Resources

- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Excellent visual guide
- [Keras Transformer Tutorial](https://keras.io/examples/nlp/text_classification_with_transformer/) - Official Keras implementation
