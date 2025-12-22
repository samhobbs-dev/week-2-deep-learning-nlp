"""
Exercise 04: Complete Production Model

Put everything together: Transformer + Regularization + Callbacks
Uses the IMDB sentiment dataset for real training.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import numpy as np
import os

print("=" * 50)
print("Exercise 04: Complete Production Model")
print("=" * 50)

# ============================================================================
# Load IMDB Dataset
# ============================================================================
vocab_size = 10000
maxlen = 200  # Longer sequences capture more context
num_classes = 2
embed_dim = 32  # Smaller embedding = less overfitting

print("\nLoading IMDB dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad sequences to fixed length
X_train = keras.utils.pad_sequences(X_train, maxlen=maxlen, truncating='post', padding='post')
X_test = keras.utils.pad_sequences(X_test, maxlen=maxlen, truncating='post', padding='post')

# Use validation set
X_val, y_val = X_test[:5000], y_test[:5000]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ============================================================================
# Build Production Model
# ============================================================================
def build_production_model():
    """Build a production-ready Transformer."""
    
    inputs = keras.Input(shape=(maxlen,))
    
    # ========================================================================
    # TODO 1: Embedding + Position (smaller dimension to reduce overfitting)
    # ========================================================================
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    positions = tf.range(maxlen)
    x = x + layers.Embedding(maxlen, embed_dim)(positions)
    
    # ========================================================================
    # TODO 2: Add dropout after embeddings (0.3 - higher for regularization)
    # ========================================================================
    # x = ???
    
    # ========================================================================
    # TODO 3: Add MultiHeadAttention with dropout
    #
    # Hint: layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.2)
    # Note: Fewer heads (2 instead of 4) reduces model capacity
    # ========================================================================
    attn = None  # TODO: Create and apply attention
    x = layers.LayerNormalization()(x + attn) if attn else x
    
    # TODO: Add dropout after attention (0.2)
    
    # ========================================================================
    # TODO 4: Add FFN with L2 regularization and dropout
    #
    # Hint:
    #   ffn = layers.Dense(64, activation='relu', 
    #                      kernel_regularizer=regularizers.l2(0.02))(x)
    #   ffn = layers.Dropout(0.3)(ffn)
    #   ffn = layers.Dense(embed_dim, 
    #                      kernel_regularizer=regularizers.l2(0.02))(ffn)
    # ========================================================================
    # TODO: Add FFN here
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)  # Higher dropout before output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build and compile
model = build_production_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================================
# TODO 5: Create callbacks (checkpoint + early stopping + LR reduction)
#
# Hint:
#   os.makedirs('checkpoints', exist_ok=True)
#   checkpoint = callbacks.ModelCheckpoint(
#       'checkpoints/best.keras', 
#       save_best_only=True,
#       monitor='val_accuracy'
#   )
#   early_stop = callbacks.EarlyStopping(
#       patience=3, 
#       restore_best_weights=True,
#       monitor='val_accuracy'
#   )
#   lr_reducer = callbacks.ReduceLROnPlateau(
#       monitor='val_loss',
#       factor=0.5,
#       patience=2,
#       min_lr=1e-5
#   )
# ============================================================================

my_callbacks = []  # TODO: Add callbacks

# ============================================================================
# Train
# ============================================================================
print("\nTraining on IMDB sentiment data...")
if my_callbacks:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=my_callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    train_loss, train_acc = model.evaluate(X_train[:5000], y_train[:5000], verbose=0)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Generalization Gap: {train_acc - val_acc:.4f}")
    
    print("\n✓ Exercise 04 complete!")
    print("Check 'checkpoints/' folder for saved model")
else:
    print("\n✗ Complete TODO 5 (callbacks) to start training")
