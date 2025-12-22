# Transformers Best Practices

## Learning Objectives
- Apply learning rate warmup schedules for stable Transformer training
- Understand when to use pre-trained models vs training from scratch
- Configure dropout and regularization for Transformers
- Implement model checkpointing for long training runs
- Follow production-ready training patterns

## Why This Matters

Transformers are powerful but require careful training. Understanding best practices matters because:

- **Training stability**: Transformers can be sensitive to hyperparameters
- **Computational efficiency**: Proper techniques reduce wasted GPU hours
- **Production deployment**: Models need to be robust and reproducible
- **Transfer learning**: Knowing when to fine-tune vs train from scratch
- **Real-world success**: The difference between "works in a notebook" and "works in production"

This reading synthesizes everything from this week: combining Transformer architecture with regularization techniques for robust, production-ready models.

## Learning Rate Warmup

### The Problem

Large learning rates at the start of training can destabilize Transformers:

```
Epoch 1 with high LR:
- Random weights produce random attention patterns
- Large gradients from random attention
- Weight updates too aggressive → training explodes

Loss: 2.3 → 5.7 → 12.4 → NaN (diverged!)
```

### The Solution: Warmup

**Gradually increase** the learning rate from 0 to target, then decay:

```
LR Schedule:
  │
  │           Peak LR
  │           /\
  │          /  \
  │         /    \  Decay
  │        /      \ 
  │       /        \
  │______/          \_______
  └─────────────────────────→
    Warmup   Training epochs
```

### Implementation in Keras

```python
import tensorflow as tf
from tensorflow import keras

class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Usage
learning_rate = WarmupSchedule(d_model=512, warmup_steps=4000)
optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)
```

### Simpler Alternative: Linear Warmup + Decay

```python
# Linear warmup then constant
warmup_epochs = 5
total_epochs = 100
peak_lr = 1e-4

def warmup_decay(epoch):
    if epoch < warmup_epochs:
        return peak_lr * epoch / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return peak_lr * 0.5 * (1 + tf.math.cos(np.pi * progress))

lr_scheduler = keras.callbacks.LearningRateScheduler(warmup_decay)

model.fit(
    x_train, y_train,
    epochs=total_epochs,
    callbacks=[lr_scheduler]
)
```

## Pre-trained Models vs Training from Scratch

### Decision Framework

```
Training Data Size:
  < 1,000 samples    → Fine-tune pre-trained (frozen base)
  1,000-100,000      → Fine-tune pre-trained (unfreeze top layers)
  > 100,000          → Consider training from scratch
  
Task Similarity:
  Similar to pre-train task  → Transfer learning
  Very different domain      → Train from scratch or domain-adapt
```

### Fine-tuning Workflow

```python
from transformers import TFBertModel, BertTokenizer

# 1. Load pre-trained model
base_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 2. Freeze base layers initially
base_model.trainable = False

# 3. Add task-specific head
inputs = keras.Input(shape=(128,), dtype='int32')
bert_output = base_model(inputs)[0]  # (batch, seq, 768)
pooled = keras.layers.GlobalAveragePooling1D()(bert_output)
outputs = keras.layers.Dense(2, activation='softmax')(pooled)

model = keras.Model(inputs, outputs)

# 4. Train head only
model.compile(optimizer=keras.optimizers.Adam(1e-4), 
              loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=5)

# 5. Unfreeze and fine-tune entire model with low LR
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Lower LR!
              loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

### When to Train from Scratch

- **Domain-specific vocabulary**: Medical, legal, code
- **Non-English languages**: If no good pre-trained model exists
- **Unique architecture needs**: Custom attention patterns
- **Sufficient data**: 100K+ examples in your domain

## Dropout Configuration for Transformers

### Where to Apply Dropout

```
Transformer Encoder Block:
  Input
    ↓
  [Multi-Head Attention]
    ↓
  [Attention Dropout: 0.1]  ← Dropout after attention
    ↓
  [Add & LayerNorm]
    ↓
  [FFN Layer 1: Dense + GELU]
    ↓
  [FFN Dropout: 0.1]        ← Dropout in FFN
    ↓
  [FFN Layer 2: Dense]
    ↓
  [Residual Dropout: 0.1]   ← Dropout before residual add
    ↓
  [Add & LayerNorm]
    ↓
  Output
```

### Standard Configurations

| Model Size | Attention Dropout | FFN Dropout | Residual Dropout |
|------------|------------------|-------------|------------------|
| Small (6 layers) | 0.1 | 0.1 | 0.1 |
| Base (12 layers) | 0.1 | 0.1 | 0.1 |
| Large (24 layers) | 0.1-0.2 | 0.1 | 0.1 |

### Implementation

```python
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, 
                 attention_dropout=0.1,
                 ffn_dropout=0.1,
                 residual_dropout=0.1):
        super().__init__()
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attention_dropout
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(ffn_dropout),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.residual_dropout = layers.Dropout(residual_dropout)
    
    def call(self, x, training=False):
        # Self-attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.residual_dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.residual_dropout(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x
```

## Model Checkpointing

### Why Checkpointing Matters

Transformer training can take hours to days. Checkpointing allows:
- Resume from interruptions
- Save best models during training
- Recover from overfitting

### Checkpoint Configuration

```python
# Save best model based on validation loss
checkpoint_best = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Save every epoch for recovery
checkpoint_every = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    save_weights_only=False
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=50,
    callbacks=[checkpoint_best, checkpoint_every]
)
```

### Loading Checkpoints

```python
# Load saved model
model = keras.models.load_model('checkpoints/best_model.keras')

# Continue training
model.fit(x_train, y_train, epochs=10, initial_epoch=50)
```

## Complete Training Pipeline

### Production-Ready Example

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
import numpy as np

def build_transformer_classifier(
    vocab_size,
    seq_length,
    num_classes,
    d_model=256,
    num_heads=4,
    num_layers=4,
    ff_dim=512,
    dropout=0.1
):
    # Input
    inputs = keras.Input(shape=(seq_length,))
    
    # Embedding + positional encoding
    x = layers.Embedding(vocab_size, d_model)(inputs)
    positions = tf.range(seq_length)
    pos_embed = layers.Embedding(seq_length, d_model)(positions)
    x = x + pos_embed
    x = layers.Dropout(dropout)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # FFN
        ffn_output = layers.Dense(ff_dim, activation='gelu')(x)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build model
model = build_transformer_classifier(
    vocab_size=10000,
    seq_length=256,
    num_classes=5
)

# Compile with warmup scheduler
lr_schedule = WarmupSchedule(d_model=256, warmup_steps=1000)
optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9,
    clipnorm=1.0  # Gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training callbacks
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
        patience=10,
        restore_best_weights=True
    ),
    # TensorBoard
    callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    ),
    # Reduce LR on plateau (after warmup)
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

# Train
history = model.fit(
    x_train, y_train,
    validation_split=0.15,
    batch_size=32,
    epochs=100,
    callbacks=training_callbacks
)
```

## Hyperparameter Guidelines

### Model Size

| Aspect | Small | Base | Large |
|--------|-------|------|-------|
| Layers | 4-6 | 12 | 24+ |
| d_model | 256 | 512-768 | 1024+ |
| Heads | 4 | 8 | 16 |
| FFN dim | 1024 | 2048-3072 | 4096+ |

### Training

| Hyperparameter | Recommended Range |
|----------------|-------------------|
| Batch size | 16-128 (limited by memory) |
| Learning rate | 1e-4 to 5e-4 (after warmup) |
| Warmup steps | 1-10% of total steps |
| Dropout | 0.1 (can increase for small data) |
| Weight decay | 0.01-0.1 |
| Gradient clip | 1.0 |

## Common Mistakes to Avoid

### 1. No Learning Rate Warmup

```python
# WRONG: High LR from start
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# RIGHT: Warmup schedule
lr_schedule = WarmupSchedule(d_model=512, warmup_steps=4000)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 2. Too High Dropout

```python
# WRONG: Excessive dropout kills gradients
layers.Dropout(0.5)  # Too high for Transformers

# RIGHT: Moderate dropout
layers.Dropout(0.1)  # Standard for Transformers
```

### 3. No Gradient Clipping

```python
# WRONG: Exploding gradients possible
optimizer = keras.optimizers.Adam()

# RIGHT: Clip gradients
optimizer = keras.optimizers.Adam(clipnorm=1.0)
```

### 4. Forgetting LayerNorm

```python
# WRONG: BatchNorm in Transformer
x = layers.BatchNormalization()(x)

# RIGHT: LayerNorm
x = layers.LayerNormalization(epsilon=1e-6)(x)
```

## Key Takeaways

1. **Learning rate warmup** is essential for stable Transformer training
2. **Fine-tuning pre-trained models** is usually better than training from scratch
3. **Dropout 0.1** is standard; increase only for small datasets
4. **Gradient clipping** prevents training explosions
5. **LayerNorm** (not BatchNorm) for Transformers
6. **Checkpoint regularly** for long training runs
7. **Early stopping** prevents overfitting
8. **TensorBoard** helps monitor training progress

## External Resources

- [Training Transformers Efficiently](https://huggingface.co/docs/transformers/perf_train_gpu_one) - HuggingFace guide
- [BERT Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training) - Complete walkthrough
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - When to apply LayerNorm
