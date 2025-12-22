"""
Exercise 03: Add Regularization

Complete the TODO sections to add regularization.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import numpy as np

print("=" * 50)
print("Exercise 03: Add Regularization")
print("=" * 50)

# Sample data (noisy, easy to overfit)
np.random.seed(42)
X = np.random.randn(500, 20).astype('float32')
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype('float32')

X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

def build_regularized_model():
    """Build a model with regularization."""
    
    model = keras.Sequential([
        # ====================================================================
        # TODO 1: Add L2 regularization to this layer
        #
        # Hint: kernel_regularizer=regularizers.l2(0.01)
        # ====================================================================
        layers.Dense(64, activation='relu'),  # TODO: Add L2
        
        # ====================================================================
        # TODO 2: Add Dropout layer (30%)
        #
        # Hint: layers.Dropout(0.3)
        # ====================================================================
        # TODO: Add dropout here
        
        layers.Dense(32, activation='relu'),
        
        # TODO: Add another dropout here (20%)
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Build model
model = build_regularized_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ============================================================================
# TODO 3: Create early stopping callback
#
# Hint:
#   early_stop = callbacks.EarlyStopping(
#       monitor='val_loss',
#       patience=5,
#       restore_best_weights=True
#   )
# ============================================================================

early_stop = None  # TODO: Create early stopping

# Train
print("\nTraining...")
if early_stop:
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val),
                       epochs=50, 
                       callbacks=[early_stop],
                       verbose=1)
    print("\n✓ Exercise 03 complete!")
else:
    print("\n✗ Complete TODO 3 (early stopping)")
