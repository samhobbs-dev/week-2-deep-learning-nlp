# Model Checkpoints

## Learning Objectives
- Understand the ModelCheckpoint callback and its importance for long training runs
- Learn checkpoint strategies: epoch-based, metric-based, and time-based
- Master saving best models automatically during training
- Apply checkpoint recovery to resume interrupted training

## Why This Matters

Model checkpointing is essential for robust machine learning workflows. Training deep learning models can take hours or days—without checkpoints, a single failure loses all progress. Understanding checkpointing is critical because:

- **Fault tolerance**: Hardware failures, power outages, or cloud interruptions won't destroy days of training
- **Best model selection**: Automatically save the best-performing model, even if training continues past the optimal point
- **Resource efficiency**: Resume training from checkpoints instead of starting over
- **Experimentation**: Save intermediate models to analyze training dynamics
- **Production safety**: Validate checkpoints before deploying to production

As you prepare models for production this week, checkpointing ensures your training time translates into reliable, recoverable model artifacts. This is a fundamental practice in professional ML workflows.

## The ModelCheckpoint Callback

### Basic Usage

```python
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# Define checkpoint callback
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}.h5',
    save_weights_only=False,  # Save entire model (architecture + weights)
    save_best_only=False,     # Save at every epoch
    verbose=1
)

# Train with checkpoint callback
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Result: Files saved at each epoch
# checkpoints/model_epoch_01.h5
# checkpoints/model_epoch_02.h5
# ...
# checkpoints/model_epoch_50.h5
```

### Key Parameters

```python
ModelCheckpoint(
    filepath='model.h5',           # Path to save model (can include variables)
    monitor='val_loss',             # Metric to monitor
    save_best_only=False,           # If True, only save when metric improves
    save_weights_only=False,        # If True, save only weights (not architecture)
    mode='auto',                    # 'min', 'max', or 'auto' (infer from metric)
    verbose=0,                      # 0 = silent, 1 = print when saving
    save_freq='epoch',              # 'epoch' or integer (batch-based)
    options=None                    # SaveOptions for SavedModel format
)
```

## Checkpoint Strategies

### 1. Save Best Model Only

Most common strategy: save only when validation metric improves.

```python
checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',         # Track validation accuracy
    save_best_only=True,            # Only save when val_accuracy improves
    mode='max',                     # Maximize accuracy (use 'min' for loss)
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Result: Only the best-performing model is saved
# Automatically overwrites when a better model is found
```

**Output during training:**

```
Epoch 1/100
...
val_accuracy: 0.85
Epoch 1: val_accuracy improved from -inf to 0.85000, saving model to models/best_model.h5

Epoch 2/100
...
val_accuracy: 0.83
Epoch 2: val_accuracy did not improve from 0.85000

Epoch 3/100
...
val_accuracy: 0.87
Epoch 3: val_accuracy improved from 0.85000 to 0.87000, saving model to models/best_model.h5
```

### 2. Monitor Validation Loss

```python
checkpoint = ModelCheckpoint(
    filepath='models/best_model_loss.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',                # Minimize loss
    verbose=1
)
```

**When to use:**
- **Classification**: `val_accuracy` (max) or `val_loss` (min)
- **Regression**: `val_loss` (min) or `val_mae` (min)
- **Custom metrics**: Specify the metric name

### 3. Save at Every Epoch (with Versioning)

```python
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.h5',
    save_best_only=False,      # Save every epoch
    verbose=1
)

# Result: Versioned checkpoints
# checkpoints/model_epoch_01_val_acc_0.85.h5
# checkpoints/model_epoch_02_val_acc_0.87.h5
# checkpoints/model_epoch_03_val_acc_0.86.h5
```

**Advantages:**
- Full training history preserved
- Can analyze model evolution
- Rollback to any epoch

**Disadvantages:**
- Large disk usage (100 epochs = 100 model files)
- Manual cleanup required

### 4. Periodic Checkpoints (Every N Epochs)

```python
# Custom callback for periodic saving
class PeriodicCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, period=5):
        super(PeriodicCheckpoint, self).__init__()
        self.filepath = filepath
        self.period = period
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            save_path = self.filepath.format(epoch=epoch+1)
            self.model.save(save_path)
            print(f"\nSaved periodic checkpoint: {save_path}")

# Save every 5 epochs
periodic_checkpoint = PeriodicCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:03d}.h5',
    period=5
)

model.fit(X_train, y_train, epochs=50, callbacks=[periodic_checkpoint])

# Result: checkpoints at epochs 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
```

### 5. Weights-Only Checkpoints

Save only weights (smaller files, faster):

```python
checkpoint = ModelCheckpoint(
    filepath='checkpoints/weights_epoch_{epoch:02d}.h5',
    save_weights_only=True,    # Only save weights
    save_best_only=False,
    verbose=1
)

# To load:
model = build_model()  # Rebuild architecture
model.load_weights('checkpoints/weights_epoch_10.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Combining Multiple Checkpoint Strategies

```python
# Strategy 1: Save best model based on validation accuracy
best_checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Strategy 2: Save periodic checkpoints for recovery
periodic_checkpoint = PeriodicCheckpoint(
    filepath='checkpoints/backup_epoch_{epoch:03d}.h5',
    period=10  # Every 10 epochs
)

# Train with both callbacks
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[best_checkpoint, periodic_checkpoint]
)

# Result:
# - models/best_model.h5 (best performing model)
# - checkpoints/backup_epoch_010.h5, backup_epoch_020.h5, ... (recovery points)
```

## Resuming Training from Checkpoint

### Load and Continue Training

```python
# Initial training
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/ongoing_training.h5', save_best_only=False)
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

# Training interrupted! Resume from checkpoint:
loaded_model = keras.models.load_model('models/ongoing_training.h5')

# Continue training (model remembers optimizer state)
loaded_model.fit(
    X_train, y_train,
    initial_epoch=10,      # Start from epoch 10
    epochs=20,             # Train to epoch 20 (10 more epochs)
    callbacks=[checkpoint]
)
```

### Track Epoch Number

```python
import json

class TrainingStateCallback(keras.callbacks.Callback):
    def __init__(self, state_file='training_state.json'):
        super(TrainingStateCallback, self).__init__()
        self.state_file = state_file
    
    def on_epoch_end(self, epoch, logs=None):
        state = {
            'last_epoch': epoch + 1,
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0))
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

# Usage
state_callback = TrainingStateCallback()
checkpoint = ModelCheckpoint('models/checkpoint.h5')

model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, state_callback]
)

# Resume training
with open('training_state.json', 'r') as f:
    state = json.load(f)

loaded_model = keras.models.load_model('models/checkpoint.h5')
loaded_model.fit(
    X_train, y_train,
    initial_epoch=state['last_epoch'],
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, state_callback]
)
```

## Checkpoint Management

### Automatic Cleanup (Keep Only Best N)

```python
import os
import glob

class BestNCheckpoints(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', keep_top_n=3):
        super(BestNCheckpoints, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.keep_top_n = keep_top_n
        self.checkpoints = []  # [(metric_value, filepath), ...]
    
    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        save_path = self.filepath.format(epoch=epoch+1, **logs)
        
        # Save current model
        self.model.save(save_path)
        self.checkpoints.append((current_metric, save_path))
        
        # Sort by metric (ascending for min, descending for max)
        reverse = (self.mode == 'max')
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Keep only top N
        if len(self.checkpoints) > self.keep_top_n:
            # Remove worst checkpoint
            _, path_to_remove = self.checkpoints.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                print(f"\nRemoved checkpoint: {path_to_remove}")

# Usage: Keep only top 3 models
best_n_checkpoint = BestNCheckpoints(
    filepath='checkpoints/model_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5',
    monitor='val_loss',
    mode='min',
    keep_top_n=3
)

model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[best_n_checkpoint])
```

### Checkpoint Rotation

```python
def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5):
    """Keep only the most recent N checkpoints"""
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, '*.h5')),
        key=os.path.getmtime,  # Sort by modification time
        reverse=True
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[keep_last_n:]:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {checkpoint}")

# Run after training or periodically
cleanup_old_checkpoints('checkpoints/', keep_last_n=5)
```

## SavedModel Format Checkpoints

### Using SavedModel Instead of H5

```python
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}',  # Directory, not .h5
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Saves as SavedModel directories:
# checkpoints/model_epoch_01/
# checkpoints/model_epoch_05/
# ...
```

**Advantages:**
- Production-ready format
- Better compatibility with TensorFlow Serving
- Recommended for deployment

**Disadvantages:**
- Larger file size (directory structure)
- More files to manage

## Practical Example: Complete Training Pipeline

```python
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Build model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
    keras.layers.LSTM(64, dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
# 1. Save best model
best_model_checkpoint = ModelCheckpoint(
    filepath='models/best_sentiment_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 2. Periodic backups
periodic_checkpoint = PeriodicCheckpoint(
    filepath='checkpoints/backup_epoch_{epoch:03d}.h5',
    period=10
)

# 3. Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 4. Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train with all callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[best_model_checkpoint, periodic_checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Load best model for evaluation
best_model = keras.models.load_model('models/best_sentiment_model.h5')
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"\nBest model test accuracy: {test_accuracy:.4f}")
```

## Cloud Storage Integration

### Save to Google Cloud Storage

```python
from tensorflow.io import gfile

checkpoint = ModelCheckpoint(
    filepath='gs://my-bucket/models/checkpoint_epoch_{epoch:02d}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# TensorFlow automatically handles GCS paths
model.fit(X_train, y_train, epochs=50, callbacks=[checkpoint])

# Load from GCS
loaded_model = keras.models.load_model('gs://my-bucket/models/checkpoint_epoch_15.h5')
```

### Save to AWS S3 (with boto3)

```python
import boto3

class S3Checkpoint(keras.callbacks.Callback):
    def __init__(self, bucket_name, s3_path, monitor='val_loss', mode='min'):
        super(S3Checkpoint, self).__init__()
        self.bucket_name = bucket_name
        self.s3_path = s3_path
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.s3_client = boto3.client('s3')
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Check if current is better than best
        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best):
            self.best = current
            
            # Save locally
            local_path = 'temp_model.h5'
            self.model.save(local_path)
            
            # Upload to S3
            s3_key = f"{self.s3_path}/model_epoch_{epoch:02d}.h5"
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"\nUploaded model to s3://{self.bucket_name}/{s3_key}")

# Usage
s3_checkpoint = S3Checkpoint(bucket_name='my-ml-models', s3_path='sentiment-classifier')
model.fit(X_train, y_train, epochs=50, callbacks=[s3_checkpoint])
```

## Key Takeaways

1. **ModelCheckpoint callback** automates model saving during training
2. **`save_best_only=True`** saves only when validation metric improves (most common strategy)
3. **Monitor `val_accuracy` (max) or `val_loss` (min)** depending on your task
4. **Periodic checkpoints** provide recovery points for long training runs
5. **Combine strategies**: Best model + periodic backups for robustness
6. **Resume training** with `initial_epoch` parameter after loading checkpoint
7. **Manage disk space**: Keep only top N checkpoints or recent N checkpoints
8. **SavedModel format** recommended for production deployment
9. **Track training state** (epoch, metrics) to resume seamlessly
10. **Cloud integration**: Save checkpoints to GCS/S3 for distributed training

## External Resources

- [Keras ModelCheckpoint Documentation](https://keras.io/api/callbacks/model_checkpoint/) - Official API reference
- [Training Checkpoints Guide](https://www.tensorflow.org/guide/checkpoint) - TensorFlow official guide
- [MLOps Best Practices](https://ml-ops.org/content/model-training) - Checkpointing in production workflows

