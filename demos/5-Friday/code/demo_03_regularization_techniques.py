"""
Demo 03: Regularization Made Simple

Prevent overfitting in 3 easy techniques:
1. Dropout
2. Weight Decay (L2)
3. Early Stopping

Quick and practical!
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import numpy as np

print("=" * 60)
print("REGULARIZATION: Prevent Your Model from Memorizing")
print("=" * 60)

print("""
THE PROBLEM: Overfitting

  Training accuracy:   99%
  Validation accuracy: 70%

  Your model MEMORIZED the training data instead of LEARNING patterns!

THE SOLUTION: Regularization

  Three easy techniques:
  1. Dropout - randomly turn off neurons
  2. Weight Decay - keep weights small
  3. Early Stopping - stop before overfitting
""")

# ============================================================================
# PART 1: Dropout
# ============================================================================

print("\n" + "=" * 60)
print("TECHNIQUE 1: Dropout")
print("=" * 60)

print("""
Dropout: Randomly "turn off" neurons during training

  Training: 20% of neurons randomly set to 0
  Inference: All neurons active (scaled appropriately)

Why it works:
  - Prevents neurons from co-adapting
  - Forces network to be redundant
  - Acts like training many smaller networks
""")

# Example with dropout
model_with_dropout = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # 30% dropout
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # 20% dropout
    layers.Dense(10, activation='softmax')
])

print("Model with Dropout:")
print("  Dense(128) -> Dropout(0.3) -> Dense(64) -> Dropout(0.2) -> Dense(10)")
print("\nRule of thumb: Dropout = 0.1-0.3 for Transformers, 0.3-0.5 for MLPs")

# ============================================================================
# PART 2: Weight Decay (L2)
# ============================================================================

print("\n" + "=" * 60)
print("TECHNIQUE 2: Weight Decay (L2 Regularization)")
print("=" * 60)

print("""
Weight Decay: Add penalty for large weights

  Loss = Data Loss + lambda * (sum of squared weights)

Why it works:
  - Large weights often mean overfitting
  - Penalty encourages smaller, more generalizable weights
""")

# Example with L2
model_with_l2 = keras.Sequential([
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01)),  # lambda = 0.01
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')
])

print("Model with L2 Regularization:")
print("  kernel_regularizer=regularizers.l2(0.01)")
print("\nRule of thumb: lambda = 0.001 to 0.01 typically")

# ============================================================================
# PART 3: Early Stopping
# ============================================================================

print("\n" + "=" * 60)
print("TECHNIQUE 3: Early Stopping")
print("=" * 60)

print("""
Early Stopping: Stop training when validation loss stops improving

  Epoch 1:  val_loss = 0.8
  Epoch 5:  val_loss = 0.3  (improving!)
  Epoch 10: val_loss = 0.25 (still improving)
  Epoch 15: val_loss = 0.28 (getting worse! STOP HERE)

The callback:
""")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,              # Wait 5 epochs before stopping
    restore_best_weights=True  # Go back to best model
)

print("""
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(x, y, callbacks=[early_stopping])
""")

# ============================================================================
# PART 4: Combining All Three
# ============================================================================

print("\n" + "=" * 60)
print("PUTTING IT TOGETHER")
print("=" * 60)

def build_regularized_model():
    return keras.Sequential([
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

print("""
Production-ready model with all techniques:

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=100,
          callbacks=[early_stopping])
""")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. DROPOUT: layers.Dropout(0.2-0.5)
   - Randomly zero neurons during training
   
2. WEIGHT DECAY: kernel_regularizer=l2(0.01)
   - Penalize large weights
   
3. EARLY STOPPING: callbacks.EarlyStopping(patience=5)
   - Stop when validation stops improving

USE ALL THREE for robust models!

Next: Combine everything into a production Transformer.
""")
