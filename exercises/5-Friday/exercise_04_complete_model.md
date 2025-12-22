# Exercise 04: Complete Production Model

## Overview
**Time:** 45 minutes  
**Type:** Implementation  
**Prerequisites:** Demo 01-04

## Learning Objective
Build a production-ready Transformer with all best practices.

## Instructions

1. Open `starter_code/exercise_04_starter.py`
2. Build the complete model combining:
   - Embedding + position encoding
   - MultiHeadAttention with dropout
   - FFN with L2 regularization
   - Model checkpointing and early stopping

3. Train and save the best model

## Hints
```python
# Attention with dropout
attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)

# FFN with regularization
ffn = layers.Dense(128, activation='relu', 
                   kernel_regularizer=regularizers.l2(0.01))(x)
ffn = layers.Dropout(0.2)(ffn)

# Callbacks
callbacks_list = [
    callbacks.ModelCheckpoint('best.keras', save_best_only=True),
    callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]
```

## Definition of Done
- [ ] Model includes all regularization techniques
- [ ] Checkpointing saves best model
- [ ] Early stopping works correctly
- [ ] Model trains without errors
