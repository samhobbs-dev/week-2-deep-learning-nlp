# Weekly Cheatsheet: Deep-Learning-NLP


## Weekly Overview

This week covered _____. By the end, trainees can:
- _____
- _____
- _____
- _____

---

## Concept Quick Reference

| Concept | Definition | Key Use Case |
|---------|------------|--------------|
| TensorBoard | _____ | _____ |
| Autoencoder | _____ | _____ |
| Reconstruction Loss | _____ | _____ |
| Latent Space | _____ | _____ |
| Backpropagation | _____ | _____ |
| Gradient Descent | _____ | _____ |
| Learning Rate | _____ | _____ |
| Batch Normalization | _____ | _____ |
| Tokenization | _____ | _____ |
| BPE (Byte-Pair Encoding) | _____ | _____ |
| One-Hot Encoding | _____ | _____ |
| Word Embedding | _____ | _____ |
| Word2Vec (Skip-gram/CBOW) | _____ | _____ |
| RNN | _____ | _____ |
| LSTM | _____ | _____ |
| GRU | _____ | _____ |
| Vanishing Gradient | _____ | _____ |
| Sequence Masking | _____ | _____ |
| Early Stopping | _____ | _____ |
| Dropout | _____ | _____ |
| L1 Regularization | _____ | _____ |
| L2 Regularization | _____ | _____ |
| Data Augmentation | _____ | _____ |
| Attention Mechanism | _____ | _____ |
| Self-Attention | _____ | _____ |
| Query (Q) | _____ | _____ |
| Key (K) | _____ | _____ |
| Value (V) | _____ | _____ |
| Multi-Head Attention | _____ | _____ |
| Transformer | _____ | _____ |
| Positional Encoding | _____ | _____ |
| Layer Normalization | _____ | _____ |
| Residual Connection | _____ | _____ |
| Feed-Forward Network (FFN) | _____ | _____ |

---

## Pros & Cons

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD | _____ | _____ | _____ |
| Adam | _____ | _____ | _____ |
| RMSprop | _____ | _____ | _____ |

### Text Encoding Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| One-Hot Encoding | _____ | _____ | _____ |
| Word Embeddings | _____ | _____ | _____ |
| Pretrained Embeddings | _____ | _____ | _____ |

### Sequential Models

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Simple RNN | _____ | _____ | _____ |
| LSTM | _____ | _____ | _____ |
| GRU | _____ | _____ | _____ |

### Regularization Techniques

| Technique | Pros | Cons | Best For |
|-----------|------|------|----------|
| Dropout | _____ | _____ | _____ |
| L1 Regularization | _____ | _____ | _____ |
| L2 Regularization | _____ | _____ | _____ |
| Early Stopping | _____ | _____ | _____ |

### Attention vs RNN

| Aspect | RNN-based | Attention-based | Best For |
|--------|-----------|-----------------|----------|
| Long-range dependencies | _____ | _____ | _____ |
| Training speed | _____ | _____ | _____ |
| Interpretability | _____ | _____ | _____ |
| Memory efficiency | _____ | _____ | _____ |

### Transformer Architectures

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Encoder-only (BERT) | _____ | _____ | _____ |
| Decoder-only (GPT) | _____ | _____ | _____ |
| Encoder-Decoder (T5) | _____ | _____ | _____ |

---

## When to Use What

### Choosing a Text Encoding

| If you have... | And you need... | Then use... | Because... |
|----------------|-----------------|-------------|------------|
| Small vocabulary | Simplicity | _____ | _____ |
| Large vocabulary | Semantic meaning | _____ | _____ |
| Limited training data | Transfer learning | _____ | _____ |

### Choosing a Sequential Model

| If your sequences are... | And you need... | Then use... | Because... |
|--------------------------|-----------------|-------------|------------|
| Short (<10 tokens) | Simple architecture | _____ | _____ |
| Long (50+ tokens) | Long-term memory | _____ | _____ |
| Variable length | Efficient training | _____ | _____ |

### Handling Overfitting

| If you observe... | Then try... | Because... |
|-------------------|-------------|------------|
| Train acc high, val acc low | _____ | _____ |
| Large weight values | _____ | _____ |
| Validation loss increasing | _____ | _____ |

### Choosing Attention vs Transformer

| If you need... | And your constraint is... | Then use... | Because... |
|----------------|---------------------------|-------------|------------|
| Sequence-to-sequence with context | Moderate compute | _____ | _____ |
| Best performance on NLP | Sufficient GPU memory | _____ | _____ |
| Very long sequences (>512) | Memory limited | _____ | _____ |
| Real-time inference | Latency critical | _____ | _____ |

---

## Essential Commands

### TensorBoard (Monday)

```python
# Set up TensorBoard callback
tensorboard_callback = tf.keras.callbacks._____(_____='./logs')

# Launch TensorBoard (in terminal)
# tensorboard --logdir=_____
```

### Autoencoder Architecture (Monday)

```python
# Encoder
encoder = keras.Sequential([
    layers.Dense(_____, activation='_____', input_shape=(_____,)),
    layers.Dense(_____, activation='_____'),  # Latent space
])

# Decoder
decoder = keras.Sequential([
    layers.Dense(_____, activation='_____'),
    layers.Dense(_____, activation='_____'),  # Reconstruct original
])
```

### Backpropagation (Tuesday)

```python
# Manual gradient computation with GradientTape
with tf.GradientTape() as tape:
    predictions = model(x_train)
    loss = loss_fn(y_train, predictions)

gradients = tape._____(loss, model._____)
optimizer.apply_gradients(zip(gradients, model._____))
```

### Batch Normalization (Tuesday)

```python
model = keras.Sequential([
    layers.Dense(64, activation='_____'),
    layers._____(___),  # Add batch normalization
    layers.Dense(32, activation='_____'),
])
```

### Tokenization (Wednesday)

```python
# Word-level tokenization
tokenizer = _____()
tokenizer.fit_on_texts(texts)
sequences = tokenizer._____(texts)

# Access vocabulary
vocab_size = len(tokenizer._____)
word_index = tokenizer._____
```

### BPE/Subword Tokenization (Wednesday)

```python
# Byte-Pair Encoding for handling OOV words
# Breaks words into subword units
# Example: "unhappiness" → ["un", "happiness"]

# Common libraries: SentencePiece, Hugging Face tokenizers
```

### Word2Vec Models (Wednesday)

```python
# Skip-gram: Predicts _____ words from _____ word
# CBOW: Predicts _____ word from _____ words

# Word arithmetic
# king - man + woman ≈ _____
# paris - france + italy ≈ _____
```

### Word Embeddings (Wednesday)

```python
# Keras Embedding layer
model.add(layers.Embedding(
    input_dim=_____,      # Vocabulary size
    output_dim=_____,     # Embedding dimension
    input_length=_____    # Sequence length
))
```

### RNN/LSTM (Thursday)

```python
# Simple RNN
model.add(layers.SimpleRNN(units=_____, activation='_____'))

# LSTM (preferred for long sequences)
model.add(layers.LSTM(units=_____, return_sequences=_____))
```

### Sequence Padding (Thursday)

```python
from tensorflow.keras.preprocessing.sequence import _____

padded = _____(sequences, maxlen=_____, padding='_____', truncating='_____')
```

### Sequence Masking (Thursday)

```python
# Masking tells model to ignore padded values
model.add(layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    mask_zero=_____  # Treat 0 as padding
))

# Or use explicit Masking layer
model.add(layers.Masking(mask_value=_____))

### Saving & Loading Models (Friday)

```python
# Save entire model (architecture + weights)
model.save('model.h5')  # _____ format
model.save('model_dir')  # _____ format (recommended)

# Load model
loaded_model = tf.keras.models._____(_____ )

# Save/load only weights
model.save_weights('weights.h5')
model.load_weights(_____)
```

### Model Checkpoints (Friday)

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_{epoch:02d}_{_____:.2f}.h5',
    monitor='_____',
    save_best_only=_____,
    mode='_____'  # 'min' for loss, 'max' for accuracy
)

### Early Stopping (Friday)

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='_____',
    patience=_____,
    restore_best_weights=_____
)
```

### Regularization (Friday)

```python
# Dropout layer
model.add(layers.Dropout(_____))

# L2 Regularization in Dense layer
model.add(layers.Dense(64, kernel_regularizer=tf.keras.regularizers._____(_____)))
```

### Attention Mechanism (Friday)

```python
# Scaled Dot-Product Attention formula
# Attention(Q, K, V) = softmax(Q · K^T / sqrt(_____)) · V

# Self-Attention: Q, K, V all from same input
query = layers.Dense(_____)(inputs)
key = layers.Dense(_____)(inputs)
value = layers.Dense(_____)(inputs)

# Compute attention scores
scores = tf.matmul(query, key, transpose_b=_____)
scores = scores / tf.math.sqrt(tf.cast(d_model, tf.float32))
attention_weights = tf.nn._____(scores, axis=-1)
output = tf.matmul(attention_weights, _____)
```

### Multi-Head Attention (Friday)

```python
# Using Keras built-in MultiHeadAttention
attention = layers.MultiHeadAttention(
    num_heads=_____,           # Number of parallel attention heads
    key_dim=_____,             # Dimension per head (d_model / num_heads)
    dropout=_____              # Dropout rate for attention weights
)

# Self-attention: query and value are the same
output = attention(_____, _____)  # (query, value)
```

### Transformer Encoder Block (Friday)

```python
# Encoder block structure:
# Input → MultiHeadAttention → Add & Norm → FFN → Add & Norm → Output

class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(_____, activation='_____'),  # Expand (typically 4x d_model)
            layers.Dense(_____),                       # Project back to d_model
        ])
        self.layernorm1 = layers._____()
        self.layernorm2 = layers._____()
    
    def call(self, x):
        attn_output = self.attention(x, x)
        x = self.layernorm1(x + _____)  # Residual connection
        ffn_output = self.ffn(x)
        return self.layernorm2(x + _____)  # Residual connection
```

### Positional Encoding (Friday)

```python
# Why needed: Transformers have no built-in _____ information
# Solution: Add position encoding to embeddings

# Learnable positional embeddings
max_seq_len = 100
d_model = 512

# Token embedding + position embedding
token_embed = layers.Embedding(vocab_size, d_model)(inputs)
positions = tf.range(max_seq_len)
pos_embed = layers.Embedding(_____, _____)(positions)
final_embed = token_embed + _____
```

---

## Common Gotchas

| Topic | Wrong | Right |
|-------|-------|-------|
| Backprop | Not detaching gradients when needed | _____ |
| Batch Norm | Placing batch norm after activation | _____ |
| Tokenization | Not handling OOV (out-of-vocabulary) words | _____ |
| Embeddings | Using randomly initialized embeddings on small data | _____ |
| RNN | Using RNN for very long sequences | _____ |
| LSTM | Forgetting to set return_sequences for stacked LSTMs | _____ |
| Padding | Padding with zeros without masking | _____ |
| Overfitting | Adding more layers when already overfitting | _____ |
| Early Stopping | Setting patience too low | _____ |
| Saving Models | Only saving weights, not architecture | _____ |
| Attention | Not scaling dot product by √d_k | _____ |
| Multi-Head | Using wrong key_dim (should be d_model/num_heads) | _____ |
| Positional Encoding | Forgetting to add position info to embeddings | _____ |
| Transformer | Not using residual connections | _____ |
| Layer Norm | Using Batch Norm instead of Layer Norm in Transformers | _____ |
| FFN | Not expanding dimension in feed-forward (should be 4x) | _____ |

---

## Key Formulas

### Backpropagation (Chain Rule) - THE CORE FORMULA
```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

Where:
- L = _____ (loss function)
- a = _____ (activation output)
- z = _____ (weighted sum before activation)
- w = _____ (weight being updated)

Forward:  Input → z = Σ(w·x) + b → a = activation(z) → Output → Loss
Backward: Loss → ∂L/∂a → ∂a/∂z → ∂z/∂w → _____ (weight update)
```

### Gradient Descent Update
```
w_new = w_old - α × ∂L/∂w

Where:
- α = _____ (learning rate)
- ∂L/∂w = _____ (gradient of loss w.r.t. weight)
```

### Momentum Update (SGD with Momentum)
```
v_t = β × v_{t-1} + (1 - β) × _____
w_new = w_old - α × _____

Where:
- β = momentum coefficient (typically _____)
- v = velocity (accumulated gradient)
```

### L2 Regularization Loss
```
Total Loss = Original Loss + λ × Σ(w²)

Where:
- λ = _____ (regularization strength)
- Σ(w²) = _____ (sum of squared weights)
```

### Dropout (Training vs Inference)
```
Training:   output = input × mask / (1 - p)
Inference:  output = input × _____

Where:
- p = _____ (dropout probability)
- mask = random binary mask (1s and 0s)
```

### Embedding Lookup
```
Vocabulary Size: V = _____
Embedding Dim:   D = _____
Embedding Matrix: V × D

Lookup: word_index → row _____ of embedding matrix → vector of dim D
```
