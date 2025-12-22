# Exercise 02: Build a Transformer

## Overview
**Time:** 45 minutes  
**Type:** Implementation  
**Prerequisites:** Demo 01, Demo 02

## Learning Objective
Build a simple Transformer text classifier from scratch.

## Instructions

1. Open `starter_code/exercise_02_starter.py`
2. Complete the model:
   - Add embedding layer for tokens
   - Add position embeddings
   - Add MultiHeadAttention layer
   - Add classification head with pooling

3. Compile and test the model

## Hints
```python
# Embedding
x = layers.Embedding()(inputs)

# Position embedding
positions = tf.range()
x = x + layers.Embedding()(positions)

# Attention
attn = layers.MultiHeadAttention()(x, x)

# Classification
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense()(x)
```

## Definition of Done
- [ ] Model compiles successfully
- [ ] Model can make predictions on sample input
- [ ] Model summary shows expected architecture
