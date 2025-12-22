"""
Solution for Exercise 04: Complete Production Model
Uses the IMDB sentiment dataset with optimized hyperparameters.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import numpy as np
import os

# ============================================================================
# Load IMDB Dataset
# ============================================================================
vocab_size = 10000
maxlen = 200  # Longer sequences capture more context
num_classes = 2
embed_dim = 32  # Smaller embedding = less overfitting

print("Loading IMDB dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad sequences to fixed length
X_train = keras.utils.pad_sequences(X_train, maxlen=maxlen, truncating='post', padding='post')
X_test = keras.utils.pad_sequences(X_test, maxlen=maxlen, truncating='post', padding='post')

# Use more data for better generalization
X_val, y_val = X_test[:5000], y_test[:5000]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ============================================================================
# SOLUTION: Build Optimized Production Transformer
# ============================================================================
def build_production_model():
    """Build a production-ready Transformer with proper regularization."""
    
    inputs = keras.Input(shape=(maxlen,))
    
    # Token + Position Embeddings (smaller dimension)
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    positions = tf.range(maxlen)
    x = x + layers.Embedding(maxlen, embed_dim)(positions)
    x = layers.Dropout(0.3)(x)  # Higher dropout on embeddings
    
    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(
        num_heads=2,  # Fewer heads = less capacity
        key_dim=16, 
        dropout=0.2
    )(x, x)
    x = layers.LayerNormalization()(x + attn)
    x = layers.Dropout(0.2)(x)
    
    # Feed-Forward Network with stronger L2
    ffn = layers.Dense(
        64, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.02)
    )(x)
    ffn = layers.Dropout(0.3)(ffn)
    ffn = layers.Dense(
        embed_dim, 
        kernel_regularizer=regularizers.l2(0.02)
    )(ffn)
    x = layers.LayerNormalization()(x + ffn)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)  # Higher dropout before output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build and compile with label smoothing
model = build_production_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# Callbacks: Checkpoint + Early Stopping + LR Reduction
# ============================================================================
os.makedirs('checkpoints', exist_ok=True)
my_callbacks = [
    callbacks.ModelCheckpoint(
        'checkpoints/best.keras', 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    ),
    callbacks.EarlyStopping(
        patience=3, 
        restore_best_weights=True, 
        monitor='val_accuracy',
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
]

# ============================================================================
# Train
# ============================================================================
print("\nTraining on IMDB sentiment data...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,  # Larger batch = more stable gradients
    callbacks=my_callbacks,
    verbose=1
)

# ============================================================================
# Evaluate
# ============================================================================
print("\n" + "=" * 50)
print("Final Evaluation")
print("=" * 50)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Show gap between train/val to demonstrate regularization effect
train_loss, train_acc = model.evaluate(X_train[:5000], y_train[:5000], verbose=0)
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Generalization Gap: {train_acc - val_acc:.4f}")

print("\n[OK] Exercise 04 Complete!")
print("Check 'checkpoints/' folder for saved model")
