# Regularization Techniques

## Learning Objectives
- Understand overfitting and why regularization is essential
- Apply Dropout to prevent co-adaptation of neurons
- Use L1/L2 weight regularization to constrain model complexity
- Implement Batch Normalization for training stability
- Combine multiple regularization techniques effectively

## Why This Matters

Regularization is critical for building production-quality models. Understanding these techniques matters because:

- **Prevents overfitting**: Models that memorize training data fail in production
- **Improves generalization**: Well-regularized models perform better on new data
- **Enables larger models**: Regularization allows using more complex architectures
- **Industry standard**: Every production model uses some form of regularization
- **Essential for Transformers**: Modern architectures rely heavily on Dropout and Layer Norm

This week's journey from RNNs to Transformers requires understanding how to train these powerful models without overfitting.

## Understanding Overfitting

### The Problem

**Overfitting**: Model memorizes training data instead of learning generalizable patterns.

```
Signs of overfitting:
  Training accuracy: 99%
  Validation accuracy: 75%
  
  Big gap = model learned training-specific patterns
  that don't generalize to new data
```

### Why It Happens

1. **Model too complex**: Too many parameters for the amount of data
2. **Training too long**: Model starts memorizing after learning patterns
3. **Insufficient data**: Not enough examples to learn general patterns
4. **Feature noise**: Model fits to irrelevant patterns in training data

### Visual Intuition

```
Underfitting:       Good Fit:        Overfitting:
    .   .               . . .            ~~~~~
   .     .            .     .           /     \
  .   X   .          .   X   .        ./   X   \.
 .    |    .        .    |    .      /     |     \
______+______      ______+______    ~/~~~~~+~~~~~\~

(too simple)      (captures pattern)  (memorizes noise)
```

## Dropout Regularization

### The Concept

During training, **randomly set a fraction of neurons to zero**. This prevents neurons from co-adapting and forces the network to learn more robust features.

```
Training (with dropout=0.2):
  Neuron 1: [0.5] → [0.5]   (keep)
  Neuron 2: [0.3] → [0.0]   (dropped!)
  Neuron 3: [0.8] → [0.8]   (keep)
  Neuron 4: [0.2] → [0.0]   (dropped!)
  Neuron 5: [0.6] → [0.6]   (keep)
  
Inference (no dropout):
  All neurons active, outputs scaled by (1-dropout_rate)
```

### Intuition: Why Dropout Works

1. **Prevents co-adaptation**: Neurons can't rely on specific other neurons
2. **Ensemble effect**: Each training step uses a different "sub-network"
3. **Feature redundancy**: Forces multiple pathways to learn same feature

### Dropout in Keras

```python
from tensorflow import keras
from keras import layers

# Basic dropout
model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),   # Drop 50% of neurons during training
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   # Drop 30%
    layers.Dense(10, activation='softmax')
])

# Dropout in RNNs/LSTMs
model = keras.Sequential([
    layers.LSTM(
        64,
        dropout=0.2,            # Input dropout
        recurrent_dropout=0.2   # Recurrent connection dropout
    ),
    layers.Dense(1, activation='sigmoid')
])

# Dropout in Transformers (built into MultiHeadAttention)
attention = layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1  # Attention dropout
)
```

### Dropout Guidelines

| Layer Type | Recommended Rate |
|-----------|------------------|
| Input layer | 0.1-0.2 |
| Hidden layers | 0.3-0.5 |
| Final layers | 0.1-0.3 |
| Attention | 0.1 |
| Recurrent | 0.2-0.3 |

**Important**: Dropout is only active during training!

```python
# During training
output = model(x, training=True)   # Dropout active

# During inference
output = model(x, training=False)  # Dropout disabled
predictions = model.predict(x)     # Dropout disabled (automatic)
```

## L1 and L2 Regularization

### Weight Penalty

Add a penalty term to the loss function based on weight magnitudes:

```
Total Loss = Data Loss + λ × Weight Penalty

L1: Weight Penalty = Σ|w|     (sum of absolute values)
L2: Weight Penalty = Σw²      (sum of squared values)
```

### L1 Regularization (Lasso)

**Effect**: Drives some weights exactly to zero → **sparse models**

```
Use when:
- Feature selection is important
- You want interpretable sparse solutions
- Many features might be irrelevant
```

```python
from keras import regularizers

# L1 regularization
model = keras.Sequential([
    layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=regularizers.l1(0.01)  # λ = 0.01
    ),
    layers.Dense(10, activation='softmax')
])
```

### L2 Regularization (Ridge/Weight Decay)

**Effect**: Shrinks all weights toward zero (but rarely exactly zero) → **smaller weights**

```
Use when:
- All features might be relevant
- You want to prevent large weight values
- More numerically stable than L1
```

```python
# L2 regularization
model = keras.Sequential([
    layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)  # λ = 0.01
    ),
    layers.Dense(10, activation='softmax')
])
```

### Elastic Net (L1 + L2)

Combine both for balanced regularization:

```python
# L1 + L2 combined
model = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)
    ),
    layers.Dense(10, activation='softmax')
])
```

### Comparison

| Aspect | L1 | L2 |
|--------|----|----|
| Effect on weights | Sparse (many zeros) | Small (all shrunk) |
| Feature selection | Yes | No |
| Computational | Harder optimization | Easier optimization |
| Use case | Few important features | Many relevant features |

## Batch Normalization

### The Problem: Internal Covariate Shift

As weights update during training, the distribution of layer inputs changes. This makes training unstable.

```
Epoch 1: Layer input mean = 0.5, std = 1.0
Epoch 2: Layer input mean = 2.3, std = 0.4  (shifted!)
Epoch 3: Layer input mean = -0.8, std = 2.1 (shifted again!)

Each layer must constantly adapt to changing inputs!
```

### The Solution

Normalize each mini-batch to have mean ≈ 0 and variance ≈ 1:

```
For each feature in mini-batch:
  1. μ = mean(x)           # Batch mean
  2. σ² = variance(x)      # Batch variance
  3. x̂ = (x - μ) / √(σ² + ε)   # Normalize
  4. y = γx̂ + β            # Scale and shift (learnable)

γ (scale) and β (shift) are trainable parameters
that allow the layer to learn the optimal distribution
```

### Batch Normalization in Keras

```python
model = keras.Sequential([
    layers.Dense(256),
    layers.BatchNormalization(),  # Normalize before activation
    layers.Activation('relu'),
    
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10, activation='softmax')
])
```

### BatchNorm vs LayerNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes over | Batch dimension | Feature dimension |
| Dependent on batch | Yes | No |
| Good for | CNNs, MLPs | RNNs, Transformers |
| Variable sequences | Problematic | Works well |

```python
# Layer Normalization (used in Transformers)
x = layers.LayerNormalization(epsilon=1e-6)(x)

# Batch Normalization
x = layers.BatchNormalization()(x)
```

### Benefits of BatchNorm

1. **Faster training**: Higher learning rates possible
2. **Less sensitive**: To weight initialization
3. **Regularization effect**: Adds noise (batch statistics vary)
4. **Reduces internal covariate shift**: Stable layer inputs

## Combining Regularization Techniques

### Recommended Combinations

```python
from tensorflow import keras
from keras import layers, regularizers

def build_regularized_model():
    model = keras.Sequential([
        # Input layer with small dropout
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Hidden layer
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Hidden layer
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Output layer (typically no regularization)
        layers.Dense(10, activation='softmax')
    ])
    
    return model

model = build_regularized_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Regularization for RNN/LSTM

```python
model = keras.Sequential([
    layers.Embedding(10000, 128),
    
    layers.LSTM(
        64,
        dropout=0.2,
        recurrent_dropout=0.2,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001)
    ),
    
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```

### Regularization for Transformers

```python
class RegularizedTransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate  # Attention dropout
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(dropout_rate),  # FFN dropout
            layers.Dense(d_model,
                        kernel_regularizer=regularizers.l2(0.001))
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x
```

## Early Stopping (Bonus)

Stop training when validation loss stops improving:

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,        # Wait 10 epochs for improvement
    restore_best_weights=True
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[early_stopping]
)
```

## Key Takeaways

1. **Overfitting** = memorizing training data, failing on new data
2. **Dropout** randomly zeros neurons, preventing co-adaptation
3. **L1 regularization** creates sparse models (feature selection)
4. **L2 regularization** shrinks weights (weight decay)
5. **Batch Normalization** normalizes layer inputs for stable training
6. **Layer Normalization** preferred for sequences (RNN, Transformer)
7. **Combine techniques**: Use dropout + weight regularization + normalization
8. **Early stopping** prevents training too long

## External Resources

- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html) - Original dropout paper by Srivastava et al.
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167) - Ioffe and Szegedy's seminal work
- [Regularization for Deep Learning](https://www.deeplearningbook.org/contents/regularization.html) - Deep Learning Book chapter
