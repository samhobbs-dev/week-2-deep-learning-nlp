# Attention Mechanism

## Learning Objectives
- Understand why attention was developed and the problems it solves
- Learn the Query, Key, Value (Q/K/V) framework for attention
- Visualize attention weights and their interpretation
- Compare attention to RNN hidden states for sequence modeling

## Why This Matters

The attention mechanism revolutionized deep learning and is the foundation of modern NLP. Understanding attention is critical because:

- **Foundation of Transformers**: The Transformer architecture (GPT, BERT, etc.) is built entirely on attention
- **Interpretability**: Attention weights show what the model "focuses" on, making it more explainable
- **Parallelization**: Unlike RNNs, attention can process all positions simultaneously
- **Long-range dependencies**: Attention directly connects any two positions in a sequence, solving the vanishing gradient problem
- **Industry standard**: Nearly all state-of-the-art NLP models use attention

This week we've progressed from simple RNNs to LSTMs. Today, we take the next leap: attention mechanisms that form the core of modern architectures like ChatGPT and BERT.

## The Problem with Sequence-to-Sequence Models

### RNN Bottleneck

In encoder-decoder RNNs, the entire input sequence must be compressed into a single "context vector":

```
Input:  "The quick brown fox jumps over the lazy dog"
         ↓    ↓      ↓     ↓     ↓    ↓    ↓    ↓    ↓
        [RNN][RNN] [RNN] [RNN] [RNN][RNN][RNN][RNN][RNN]
                                                     ↓
                              Context Vector (fixed-size, e.g., 256-dim)
                                     ↓
                              [Decoder RNN] → Translation
```

**Problems:**
1. **Information bottleneck**: All information squeezed into one vector
2. **Long sequences suffer**: Context vector can't hold everything
3. **Early words forgotten**: By the time decoding starts, early input information has faded

## The Attention Solution

### Key Insight

Instead of using just the final hidden state, **look at all encoder hidden states** and decide which ones are relevant for each decoder step.

```
Encoder:  h₁   h₂   h₃   h₄   h₅   (all hidden states kept)
           \    \    |    /    /
            \    \   |   /    /
             ↘   ↘  ↓  ↙   ↙
           attention weights: [0.1, 0.05, 0.6, 0.2, 0.05]
                     ↓
              context = weighted sum of h₁...h₅
                     ↓
            Decoder step 1 → output "Le"
```

At each decoder step, we compute a **different weighted combination** of encoder states based on what's relevant.

## Query-Key-Value Framework

### The Analogy

Think of attention like a **database lookup**:

- **Query (Q)**: What you're looking for (decoder state)
- **Keys (K)**: Labels on each item (encoder states)
- **Values (V)**: The actual content (also encoder states, or transformed versions)

```
Student looking for a book:
  Query: "Deep learning textbook"
  Keys:  [Fiction, History, Deep Learning, Biology, Math]
  Values: [Book1,   Book2,   Book3,         Book4,   Book5]
  
  Attention weights: [0.0, 0.0, 0.95, 0.0, 0.05]  (high match for "Deep Learning")
  Output: 0.95×Book3 + 0.05×Book5
```

### Mathematical Formulation

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Where:
- Q: Query matrix (what we're looking for)
- K: Key matrix (what we match against)
- V: Value matrix (what we retrieve)
- d_k: Dimension of keys (for scaling)
- Q · K^T: Dot product measures similarity
- softmax: Converts scores to probabilities
- √d_k: Scaling factor to prevent extremely large dot products
```

### Step-by-Step Example

```python
import numpy as np

# Simple example: 3 positions, 4-dimensional vectors
d_k = 4  # Key/Query dimension

# Query: What we're looking for
Q = np.array([[1.0, 0.5, -0.5, 0.2]])  # Shape: (1, 4)

# Keys: What each position represents
K = np.array([
    [1.0, 0.4, -0.3, 0.1],   # Key 1 - similar to query
    [-0.5, 0.8, 0.2, -0.4],  # Key 2 - different
    [0.9, 0.5, -0.6, 0.3]    # Key 3 - very similar to query
])  # Shape: (3, 4)

# Values: Content at each position
V = np.array([
    [0.1, 0.2],  # Value 1
    [0.8, 0.9],  # Value 2
    [0.3, 0.4]   # Value 3
])  # Shape: (3, 2)

# Step 1: Compute attention scores
scores = Q @ K.T  # Shape: (1, 3)
print(f"Raw scores: {scores}")
# [[1.39, -0.07, 1.58]]  ← Position 3 has highest score

# Step 2: Scale by sqrt(d_k)
scaled_scores = scores / np.sqrt(d_k)
print(f"Scaled scores: {scaled_scores}")
# [[0.695, -0.035, 0.79]]

# Step 3: Apply softmax to get attention weights
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
print(f"Attention weights: {attention_weights}")
# [[0.38, 0.18, 0.44]]  ← Position 3 gets most attention

# Step 4: Weighted sum of values
output = attention_weights @ V
print(f"Output: {output}")
# [[0.26, 0.36]]  ← Weighted combination of all values
```

## Self-Attention

### Attending to Yourself

In **self-attention**, a sequence attends to itself. Each position can look at all other positions (and itself).

```
Input sentence:  "The cat sat on the mat"
                  ↓   ↓   ↓   ↓   ↓   ↓
                  
For word "sat":
  Q = transform("sat")
  K = transform(["The", "cat", "sat", "on", "the", "mat"])
  V = transform(["The", "cat", "sat", "on", "the", "mat"])
  
  Attention weights might be: [0.05, 0.4, 0.3, 0.05, 0.05, 0.15]
                                      ↑          ↑         ↑
                               "cat"  "sat"            "mat"
  
  The model learns "sat" is related to "cat" (the subject) and "mat" (where)
```

### Self-Attention in Keras

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Create input
batch_size = 2
seq_length = 5
embed_dim = 64

# Input: (batch, sequence_length, embedding_dim)
inputs = keras.Input(shape=(seq_length, embed_dim))

# Self-attention: Q, K, V all come from the same input
# Simple implementation using Dense layers
query = layers.Dense(embed_dim)(inputs)   # Q = W_Q × input
key = layers.Dense(embed_dim)(inputs)     # K = W_K × input
value = layers.Dense(embed_dim)(inputs)   # V = W_V × input

# Compute attention scores: Q × K^T / sqrt(d)
scores = tf.matmul(query, key, transpose_b=True)  # (batch, seq, seq)
scores = scores / tf.math.sqrt(tf.cast(embed_dim, tf.float32))

# Apply softmax
attention_weights = tf.nn.softmax(scores, axis=-1)

# Weighted sum of values
output = tf.matmul(attention_weights, value)  # (batch, seq, embed_dim)

model = keras.Model(inputs, output)
print(f"Output shape: {model.output_shape}")  # (None, 5, 64)
```

## Visualizing Attention Weights

### Attention Heatmap

Attention weights can be visualized as a matrix showing which positions attend to which:

```python
import matplotlib.pyplot as plt
import numpy as np

# Example attention weights for "The cat sat on the mat"
words = ["The", "cat", "sat", "on", "the", "mat"]
attention = np.array([
    [0.8, 0.1, 0.05, 0.02, 0.02, 0.01],  # "The" attends mostly to itself
    [0.1, 0.5, 0.2, 0.05, 0.05, 0.1],    # "cat" attends to itself and "sat"
    [0.05, 0.4, 0.3, 0.05, 0.05, 0.15],  # "sat" attends to "cat" and "mat"
    [0.02, 0.1, 0.3, 0.5, 0.02, 0.06],   # "on" attends to itself and "sat"
    [0.7, 0.05, 0.05, 0.02, 0.15, 0.03], # "the" attends to first "The"
    [0.02, 0.2, 0.3, 0.1, 0.02, 0.36]    # "mat" attends to "sat" and itself
])

plt.figure(figsize=(8, 6))
plt.imshow(attention, cmap='Blues')
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.xlabel("Key (attended to)")
plt.ylabel("Query (attending from)")
plt.colorbar(label="Attention Weight")
plt.title("Self-Attention Weights")
plt.show()
```

### Interpreting Attention

- **Diagonal patterns**: Words attending to themselves
- **Subject-verb connections**: "cat" and "sat" have high mutual attention
- **Coreference**: "the" (second) attends to "The" (first)

## Attention vs RNN Hidden States

| Aspect | RNN Hidden State | Attention |
|--------|-----------------|-----------|
| **Context access** | Only previous positions | All positions |
| **Long-range** | Degrades over distance | Direct connection |
| **Computation** | Sequential (slow) | Parallel (fast) |
| **Interpretability** | Opaque | Visible weights |
| **Memory** | Fixed-size vector | Weighted sum of all |

### Why Attention is Better for Long Sequences

```
RNN processing "The cat sat on the mat":
  h₁ → h₂ → h₃ → h₄ → h₅ → h₆
  Information about "The" must pass through 5 transformations to reach "mat"
  Each step loses some information

Attention processing:
  Every word can directly attend to every other word
  "mat" can directly look at "The" with one computation
  No information loss through sequential processing
```

## Key Takeaways

1. **Attention solves the bottleneck problem** by allowing access to all encoder states
2. **Query-Key-Value framework**: Q asks "what am I looking for?", K says "what do I have?", V provides the content
3. **Self-attention** allows a sequence to attend to itself, capturing relationships between all positions
4. **Attention weights are interpretable** - you can visualize what the model focuses on
5. **Parallelizable** - unlike RNNs, attention can process all positions simultaneously
6. **Foundation for Transformers** - next topic! Transformers replace RNNs entirely with attention

## External Resources

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762) - The seminal Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar's visual guide
- [Visualizing Attention in Transformers](https://github.com/jessevig/bertviz) - BertViz tool for exploring attention
