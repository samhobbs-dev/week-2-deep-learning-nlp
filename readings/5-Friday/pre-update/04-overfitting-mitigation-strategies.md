# Overfitting Mitigation Strategies

## Learning Objectives
- Understand overfitting and how to detect it in training curves
- Master regularization techniques: dropout, L1/L2 regularization
- Learn data augmentation strategies for NLP and image data
- Apply cross-validation and ensemble methods to improve generalization

## Why This Matters

Overfitting is the most common failure mode in machine learning—models that perform brilliantly on training data but fail miserably in production. Understanding overfitting mitigation is critical because:

- **Production performance**: Overfitted models deliver poor real-world results despite high training accuracy
- **Business value**: Customers receive bad predictions, destroying trust and ROI
- **Resource waste**: Time and money spent training models that don't generalize
- **Competitive advantage**: Robust models that generalize well are the difference between academic projects and production systems
- **Regulatory compliance**: In healthcare and finance, overfitting can have legal consequences

As you complete this week's deep learning and NLP fundamentals, overfitting mitigation is what separates student projects from production-ready systems. This is the final critical skill for deploying reliable models.

## Understanding Overfitting

### What is Overfitting?

**Overfitting** occurs when a model learns the training data too well, including noise and outliers, failing to generalize to new data.

```
Perfect memorization ≠ True understanding

Example:
Student memorizes exam questions and answers
→ 100% on practice exam
→ Fails real exam (different questions, same concepts)

Model memorizes training examples
→ 100% training accuracy
→ Poor test accuracy (different examples, same patterns)
```

### Detecting Overfitting

**Training vs. Validation Curves:**

```python
import matplotlib.pyplot as plt

# Healthy training (good generalization)
plt.plot([0.6, 0.4, 0.3, 0.25, 0.22], label='Train Loss')
plt.plot([0.62, 0.42, 0.32, 0.27, 0.24], label='Val Loss')
# Both decrease together, small gap

# Overfitting
plt.plot([0.6, 0.4, 0.2, 0.1, 0.05], label='Train Loss')
plt.plot([0.62, 0.42, 0.45, 0.50, 0.58], label='Val Loss')
# Train loss decreases, val loss increases → OVERFITTING!
```

**Signs of overfitting:**
1. **Training accuracy >> validation accuracy** (e.g., 95% train, 70% val)
2. **Validation loss increases** while training loss decreases
3. **Large gap** between training and validation metrics

### Causes of Overfitting

1. **Model too complex**: Too many parameters relative to data
2. **Insufficient data**: Not enough examples to learn general patterns
3. **Training too long**: Model starts memorizing noise
4. **No regularization**: Nothing prevents overfitting

## Regularization Techniques

### 1. Dropout

**Idea**: Randomly deactivate neurons during training, forcing network to learn robust features.

```python
from tensorflow.keras import layers

# Model with dropout
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    # dropout: 20% of inputs randomly dropped
    # recurrent_dropout: 20% of recurrent connections dropped
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # 50% of neurons randomly dropped
    
    layers.Dense(1, activation='sigmoid')
])
```

**How it works:**

```
Training:
Epoch 1, Sample 1: Drop neurons [3, 7, 12] → Force other neurons to compensate
Epoch 1, Sample 2: Drop neurons [1, 5, 9] → Different neurons learn
Result: Network can't rely on any single neuron, learns distributed representations

Inference:
All neurons active, outputs scaled appropriately
```

**Typical dropout rates:**
- **0.2-0.3**: Light regularization (LSTM recurrent dropout)
- **0.5**: Standard (Dense layers)
- **0.7-0.8**: Heavy regularization (rare, risk underfitting)

**Best practices:**

```python
# LSTM with dropout
layers.LSTM(
    64,
    dropout=0.2,             # Input dropout
    recurrent_dropout=0.2    # Recurrent connection dropout
)

# Dense layer with dropout
layers.Dense(128, activation='relu')
layers.Dropout(0.5)  # Apply dropout AFTER activation

# Dropout NOT applied during inference
model.predict(X_test)  # Dropout automatically disabled
```

### 2. L1 and L2 Regularization

**Idea**: Add penalty to loss function for large weights, encouraging simpler models.

**L2 Regularization (Ridge):**

```python
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)  # L2 penalty: 0.01 * sum(weights²)
    ),
    layers.Dense(1, activation='sigmoid')
])

# Loss becomes: original_loss + 0.01 * sum(weights²)
# Large weights penalized → model prefers smaller weights
```

**L1 Regularization (Lasso):**

```python
model = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l1(0.01)  # L1 penalty: 0.01 * sum(|weights|)
    ),
    layers.Dense(1, activation='sigmoid')
])

# L1 encourages sparsity (many weights become exactly 0)
```

**L1 + L2 Combined (Elastic Net):**

```python
model = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
    ),
    layers.Dense(1, activation='sigmoid')
])
```

**When to use:**
- **L2**: General-purpose regularization (most common)
- **L1**: Feature selection (forces irrelevant weights to 0)
- **L1+L2**: Best of both worlds

**Typical values:**
- Start with 0.01 or 0.001
- Increase if still overfitting (0.1)
- Decrease if underfitting (0.0001)

### 3. Batch Normalization

**Idea**: Normalize activations, reducing internal covariate shift and adding regularization effect.

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),  # Normalize activations
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(1, activation='sigmoid')
])
```

**Regularization effect:**
- Batch normalization adds noise (batch statistics vary)
- Acts as mild regularizer
- Can reduce need for dropout (though combining both is common)

## Data-Based Strategies

### 1. More Training Data

**Most effective solution**: More diverse examples → better generalization.

```python
# Insufficient data (overfitting likely)
X_train.shape  # (500, 100) — 500 examples

# Sufficient data (better generalization)
X_train.shape  # (50000, 100) — 50,000 examples
```

**When you can't get more data:**
- Use data augmentation (below)
- Transfer learning (pre-trained embeddings)
- Regularization techniques

### 2. Data Augmentation

#### For Images

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augment images during training
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotate up to 20 degrees
    width_shift_range=0.2,    # Shift horizontally by 20%
    height_shift_range=0.2,   # Shift vertically by 20%
    horizontal_flip=True,     # Random horizontal flip
    zoom_range=0.2            # Random zoom
)

# Generate augmented batches
datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50)

# Original: 10,000 images
# With augmentation: Effectively millions of variations
```

#### For Text (NLP)

```python
# 1. Synonym replacement
original = "The movie was great"
augmented = "The film was excellent"

# 2. Back-translation
original = "I love this product"
translate_to_french = "J'adore ce produit"
translate_back_to_english = "I adore this product"  # Slight variation

# 3. Random insertion/deletion/swap
original = "This is a good example"
augmented = "This is a really good example"  # Insertion
augmented = "This is good example"           # Deletion
augmented = "This is a example good"         # Swap
```

**NLP augmentation libraries:**

```python
# Using nlpaug library
import nlpaug.augmenter.word as naw

# Synonym augmentation
aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment("The movie was great")
print(augmented_text)  # "The film was excellent"

# Contextual word embeddings augmentation
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
augmented_text = aug.augment("I love NLP")
print(augmented_text)  # "I really love NLP" (BERT suggests insertion)
```

### 3. Cross-Validation

**Idea**: Train on multiple train/validation splits to ensure robust performance.

```python
from sklearn.model_selection import KFold
import numpy as np

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\nTraining fold {fold + 1}/5")
    
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Build fresh model for each fold
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_train_fold, y_train_fold, epochs=20, verbose=0)
    
    # Evaluate
    score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    scores.append(score[1])  # Accuracy
    print(f"Fold {fold + 1} accuracy: {score[1]:.4f}")

# Average performance across folds
print(f"\nMean accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# If std is high → model unstable (overfitting or data issues)
# If std is low → model robust
```

## Model Architecture Strategies

### 1. Reduce Model Complexity

**Overly complex model:**

```python
# Too many parameters for small dataset
model = keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=512),  # Large embedding
    layers.LSTM(256),                                  # Large LSTM
    layers.Dense(256, activation='relu'),              # Large dense
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
# Total params: ~2 million for 10,000 training examples → overfitting!
```

**Simpler model:**

```python
# Appropriate complexity for small dataset
model = keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=64),   # Smaller embedding
    layers.LSTM(32),                                   # Smaller LSTM
    layers.Dense(16, activation='relu'),               # Smaller dense
    layers.Dense(1, activation='sigmoid')
])
# Total params: ~50,000 → better generalization
```

**Rule of thumb**: Parameters should be < 10% of training examples.

### 2. Early Stopping

Prevent training too long:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=200,  # Set high, early stopping will interrupt
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Stops when validation loss stops improving → prevents overfitting
```

### 3. Ensemble Methods

**Idea**: Combine multiple models to reduce overfitting and variance.

**Simple averaging:**

```python
# Train multiple models with different initializations
models = []
for i in range(5):
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=20, verbose=0)
    models.append(model)

# Predict with ensemble
predictions = []
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

# Average predictions
ensemble_prediction = np.mean(predictions, axis=0)

# Ensemble typically outperforms any single model
```

**Stacking:**

```python
# Train base models
model_1 = train_model_1()
model_2 = train_model_2()
model_3 = train_model_3()

# Use base model predictions as features for meta-model
base_predictions = np.column_stack([
    model_1.predict(X_train),
    model_2.predict(X_train),
    model_3.predict(X_train)
])

# Train meta-model on base predictions
meta_model = LogisticRegression()
meta_model.fit(base_predictions, y_train)
```

## Practical Example: Mitigating Overfitting

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Baseline model (overfits)
def baseline_model():
    return keras.Sequential([
        layers.Embedding(input_dim=10000, output_dim=256),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

# Regularized model
def regularized_model():
    return keras.Sequential([
        # Smaller embedding
        layers.Embedding(input_dim=10000, output_dim=64, mask_zero=True),
        
        # LSTM with dropout
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        
        # Dense with L2 regularization and dropout
        layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        ),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ModelCheckpoint('models/best_regularized_model.h5', save_best_only=True)
]

# Train regularized model
model = regularized_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Compare training vs validation curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.tight_layout()
plt.show()

# Healthy curves: Train and val close together, both improving
```

## Checklist: Preventing Overfitting

### Before Training
- [ ] Collect more data if possible
- [ ] Split data properly (train/val/test)
- [ ] Use appropriate model complexity for dataset size

### Model Architecture
- [ ] Add dropout layers (0.2-0.5)
- [ ] Use L2 regularization (0.01-0.001)
- [ ] Consider batch normalization
- [ ] Start with simpler architecture, add complexity only if needed

### Training Process
- [ ] Use early stopping (patience 10-15)
- [ ] Monitor validation metrics, not just training metrics
- [ ] Use learning rate scheduling (ReduceLROnPlateau)
- [ ] Save best model based on validation performance

### Data Strategies
- [ ] Apply data augmentation if applicable
- [ ] Use cross-validation for robust evaluation
- [ ] Ensure balanced class distribution

### Evaluation
- [ ] Check train/val gap (should be small)
- [ ] Evaluate on held-out test set
- [ ] Consider ensemble methods for production

## Key Takeaways

1. **Overfitting = good training performance, poor validation/test performance**
2. **Detect overfitting** by monitoring train/val curves (diverging = overfitting)
3. **Dropout** randomly deactivates neurons, forcing robust representations (0.2-0.5 typical)
4. **L2 regularization** penalizes large weights, encouraging simpler models (0.01 typical)
5. **Early stopping** prevents training too long (patience 10-15 epochs)
6. **More data** is the best solution when available
7. **Data augmentation** artificially increases dataset size (especially for images)
8. **Cross-validation** ensures robust performance estimates
9. **Simpler models** generalize better than unnecessarily complex ones
10. **Ensemble methods** combine multiple models to reduce variance

## External Resources

- [Deep Learning Book - Regularization](https://www.deeplearningbook.org/contents/regularization.html) - Comprehensive theoretical treatment
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) - Original dropout paper
- [Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) - Best practices for training deep networks

