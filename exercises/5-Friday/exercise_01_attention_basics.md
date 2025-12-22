# Exercise 01: Attention Basics

## Overview
**Time:** 30 minutes  
**Type:** Implementation  
**Prerequisites:** Demo 01

## Learning Objective
Use Keras MultiHeadAttention to process a sequence.

## Instructions

1. Open `starter_code/exercise_01_starter.py`
2. Complete the TODO sections:
   - Create a `MultiHeadAttention` layer with 4 heads
   - Apply self-attention to the input sequence
3. Run the script and verify the output shape

## Hints
```python
# Create attention layer
layers.MultiHeadAttention()

# Apply self-attention (query and value are the same)
attention()
```

## Definition of Done
- [ ] Attention layer created with correct parameters
- [ ] Output shape matches input shape
- [ ] Script runs without errors
