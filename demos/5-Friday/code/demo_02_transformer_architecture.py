"""
Demo 02: Transformers Made Simple

This demo shows:
1. What a Transformer is (diagram)
2. Build one with Keras in 20 lines
3. Train on text classification

NO deep architecture details. Focus on USING transformers.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

print("=" * 60)
print("TRANSFORMERS: The Architecture Behind ChatGPT")
print("=" * 60)

print("""
What is a Transformer?

  Input Text -> [Embedding] -> [Attention Blocks] -> [Output]

The magic is in the Attention Blocks:
  - Look at ALL words simultaneously
  - Learn which words relate to each other
  - Stack multiple blocks for deeper understanding

That's it! No RNNs, no sequential processing.
""")

# ============================================================================
# PART 1: Building a Transformer Classifier
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: Build a Transformer Text Classifier")
print("=" * 60)

def build_simple_transformer(vocab_size=10000, maxlen=100, num_classes=2):
    """A simple Transformer in ~20 lines."""
    
    inputs = keras.Input(shape=(maxlen,))
    
    # Embedding layer (words to vectors)
    x = layers.Embedding(vocab_size, 64)(inputs)
    
    # Add position information (Transformers need this!)
    positions = tf.range(maxlen)
    pos_embed = layers.Embedding(maxlen, 64)(positions)
    x = x + pos_embed
    
    # Transformer block (the magic!)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1
    )(x, x)
    x = layers.LayerNormalization()(x + attention_output)
    
    # Simple feed-forward
    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(64)(ffn)
    x = layers.LayerNormalization()(x + ffn)
    
    # Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build it
model = build_simple_transformer()
print("\nModel Summary:")
model.summary()

# ============================================================================
# PART 2: Train on IMDB Sentiment
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: Quick Training Demo")
print("=" * 60)

# Load IMDB data
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

# Use small subset for fast demo
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]

print(f"Training on {len(x_train)} samples...")

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.2%}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. Transformer = Embedding + Attention Blocks + Output
2. Position embeddings are REQUIRED (attention has no order!)
3. LayerNormalization after each sub-layer
4. In Keras: MultiHeadAttention + Dense layers
5. Works great for text classification!

Key hyperparameters:
  - num_heads: 4-12 typical
  - key_dim: 16-64 typical  
  - Dropout: 0.1-0.3

This same architecture powers BERT, GPT, and more!
Next: Regularization to prevent overfitting.
""")
