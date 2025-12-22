# Exercise 03: Add Regularization

## Overview
**Time:** 30 minutes  
**Type:** Implementation  
**Prerequisites:** Demo 03

## Learning Objective
Apply regularization techniques to prevent overfitting.

## Instructions

1. Open `starter_code/exercise_03_starter.py`
2. Add three regularization techniques:
   - L2 regularization to Dense layers
   - Dropout layers between Dense layers
   - Early stopping callback

3. Train the model and observe the improvement

## Hints
```python
# L2 regularization
layers.Dense()

# Dropout
layers.Dropout()

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor,
    patience,
    restore_best_weights
)
```

## Definition of Done
- [ ] L2 regularization applied to Dense layers
- [ ] Dropout layers added (0.2-0.3)
- [ ] Early stopping configured and triggers
- [ ] Gap between train/val accuracy reduced
