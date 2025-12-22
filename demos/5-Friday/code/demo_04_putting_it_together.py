"""
Demo 04: Complete Transformer with Best Practices

Put it all together:
1. Simple Transformer
2. With regularization
3. Proper training callbacks

One complete, production-ready example.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import numpy as np
import os

print("=" * 60)
print("COMPLETE TRANSFORMER: Production-Ready Example")
print("=" * 60)

# ============================================================================
# THE MODEL
# ============================================================================

print("\n" + "=" * 60)
print("Building Production Transformer")
print("=" * 60)

def build_production_transformer(
    vocab_size=10000,
    maxlen=100,
    num_classes=2,
    d_model=64,
    dropout=0.2
):
    """Production-ready Transformer classifier."""
    
    inputs = keras.Input(shape=(maxlen,))
    
    # Embeddings with position
    x = layers.Embedding(vocab_size, d_model)(inputs)
    positions = tf.range(maxlen)
    x = x + layers.Embedding(maxlen, d_model)(positions)
    x = layers.Dropout(dropout)(x)
    
    # Simple Transformer block
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)
    
    ffn = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(d_model, kernel_regularizer=regularizers.l2(0.01))(ffn)
    x = layers.LayerNormalization()(x + ffn)
    
    # Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

model = build_production_transformer()
print("Model built!")
model.summary()

# ============================================================================
# TRAINING WITH BEST PRACTICES
# ============================================================================

print("\n" + "=" * 60)
print("Training with All Best Practices")
print("=" * 60)

# Setup callbacks
os.makedirs('checkpoints', exist_ok=True)

training_callbacks = [
    # Save best model
    callbacks.ModelCheckpoint(
        'checkpoints/best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

print("""
Callbacks configured:
  [OK] ModelCheckpoint - saves best model
  [OK] EarlyStopping - stops when overfitting
""")

# Load data
print("\nLoading IMDB data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

# Small subset for demo
x_train, y_train = x_train[:3000], y_train[:3000]
x_test, y_test = x_test[:500], y_test[:500]

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("""
Compile options:
  [OK] Adam optimizer
  [OK] Gradient clipping (clipnorm=1.0)
""")

# Train
print("\nTraining...")
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=training_callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[OK] Test Accuracy: {test_acc:.2%}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("WHAT WE BUILT")
print("=" * 60)

print("""
Production Transformer Checklist:

[OK] Architecture:
  - Embedding + Position encoding
  - MultiHeadAttention layer
  - LayerNormalization
  - Feed-forward network

[OK] Regularization:
  - Dropout (0.2)
  - L2 weight decay (0.01)
  - Early stopping

[OK] Training:
  - Adam optimizer
  - Gradient clipping
  - Model checkpointing

This same pattern scales to BERT, GPT, etc.!
""")

print("\n" + "=" * 60)
print("WEEK 2 COMPLETE!")
print("=" * 60)

print("""
You've learned:
  Monday:    TensorBoard, Autoencoders
  Tuesday:   Backpropagation, Optimization
  Wednesday: NLP Basics, Embeddings
  Thursday:  RNNs, LSTMs
  Friday:    Attention, Transformers, Regularization

Next week: Vector Databases!
""")
