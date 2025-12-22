# Early Stopping and Callbacks

## Learning Objectives
- Understand the EarlyStopping mechanism and how it prevents overfitting
- Learn to configure patience, monitoring metrics, and weight restoration
- Master common Keras callbacks: ReduceLROnPlateau, TensorBoard, CSVLogger
- Apply callback combinations for robust training pipelines

## Why This Matters

Early stopping and callbacks are essential for efficient, production-quality training. Without them, you waste computational resources and risk deploying overfitted models. Understanding these tools is critical because:

- **Prevents overfitting**: Stops training automatically when validation performance stops improving
- **Saves time and money**: Avoids unnecessary epochs (hours or days of wasted GPU time)
- **Automation**: Reduces manual monitoring of training curves
- **Production robustness**: Callbacks handle learning rate adjustments, logging, and recovery automatically
- **Reproducibility**: Consistent training behavior across experiments

As you finalize your deep learning workflow this week, callbacks transform manual, error-prone training into automated, reliable pipelines suitable for production deployment.

## Early Stopping: Automatic Overfitting Prevention

### The Overfitting Problem

```
Training continues:
Epoch 10: val_loss = 0.25, val_accuracy = 0.90  ← Best performance
Epoch 15: val_loss = 0.26, val_accuracy = 0.89
Epoch 20: val_loss = 0.29, val_accuracy = 0.87
Epoch 25: val_loss = 0.33, val_accuracy = 0.85  ← Training loss still decreasing!

Problem: Model continues training, getting worse on validation set
Solution: Stop at epoch 10 (best validation performance)
```

### EarlyStopping Callback

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',          # Metric to monitor
    patience=10,                 # Wait 10 epochs for improvement
    restore_best_weights=True,   # Restore weights from best epoch
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=100,                  # Set high, early stopping will interrupt
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Training might stop at epoch 35 if val_loss hasn't improved for 10 epochs
```

**Output during training:**

```
Epoch 25/100
val_loss: 0.24

Epoch 30/100
val_loss: 0.23  ← Best so far

Epoch 35/100
val_loss: 0.24

Epoch 40/100
val_loss: 0.25
Restoring model weights from the end of the best epoch: 30.
Epoch 40: early stopping  ← Stopped! No improvement for 10 epochs
```

### Key Parameters

```python
EarlyStopping(
    monitor='val_loss',          # Metric to monitor ('val_loss', 'val_accuracy', etc.)
    min_delta=0,                 # Minimum change to qualify as improvement
    patience=0,                  # Number of epochs with no improvement before stopping
    verbose=0,                   # 0=silent, 1=print messages
    mode='auto',                 # 'min', 'max', or 'auto' (infer from metric name)
    baseline=None,               # Stop if metric doesn't reach this baseline
    restore_best_weights=False   # Restore weights from best epoch
)
```

### Configuring Patience

**Low patience (3-5 epochs):**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=3)
# Stops quickly, risk of premature stopping
```

**Use when:**
- Fast prototyping
- Limited computational budget
- Confident in hyperparameters

**High patience (10-20 epochs):**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=15)
# Allows more time for improvement, slower stopping
```

**Use when:**
- Production training
- Complex models that improve slowly
- Noisy validation metrics

### Minimum Delta

Require meaningful improvement:

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001  # Must improve by at least 0.001 to count
)

# Scenario:
# Epoch 50: val_loss = 0.250
# Epoch 51: val_loss = 0.2498  ← Change < 0.001, doesn't count as improvement
# Epoch 52: val_loss = 0.248   ← Change = 0.002 > 0.001, counts as improvement
```

**Prevents stopping on noise**: Small fluctuations don't reset patience counter.

### Restore Best Weights

```python
# WITHOUT restore_best_weights
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
# Result: Model ends with weights from stopping epoch (not necessarily best)

# WITH restore_best_weights
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Result: Model automatically restored to best epoch's weights
```

**Recommendation**: **Always use `restore_best_weights=True`** for production.

### Monitoring Different Metrics

```python
# Stop based on validation accuracy (maximize)
early_stop_acc = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',  # Stop if accuracy doesn't increase
    restore_best_weights=True
)

# Stop based on custom metric
early_stop_f1 = EarlyStopping(
    monitor='val_f1_score',  # Custom metric (must be in model.compile metrics)
    patience=15,
    mode='max',
    restore_best_weights=True
)
```

## Combining Early Stopping with Checkpointing

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Save best model
checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Stop training if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Train with both
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop]
)

# Result:
# - Training stops early (e.g., epoch 87)
# - Model weights restored to best epoch (e.g., epoch 72)
# - Best model saved to disk at models/best_model.h5
```

## ReduceLROnPlateau: Adaptive Learning Rate

### The Problem: Plateaus

```
Training progress:
Epoch 1-20:  val_loss decreases rapidly (0.8 → 0.4)
Epoch 21-50: val_loss plateaus (0.4 → 0.38 → 0.39 → 0.38 → ...)
             Learning rate too high to fine-tune

Solution: Reduce learning rate when validation loss stops improving
```

### ReduceLROnPlateau Callback

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,             # Multiply LR by 0.5
    patience=5,             # Wait 5 epochs before reducing
    min_lr=1e-7,            # Don't go below this LR
    verbose=1
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[reduce_lr])
```

**Output during training:**

```
Epoch 25/100
val_loss: 0.38, learning_rate: 0.001

Epoch 30/100
val_loss: 0.38
Epoch 30: ReduceLROnPlateau reducing learning rate to 0.0005  ← LR reduced!

Epoch 35/100
val_loss: 0.35  ← Improvement after LR reduction
```

### Key Parameters

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,              # New LR = old LR * factor (0.1 = reduce to 10%)
    patience=10,             # Wait 10 epochs before reducing
    verbose=0,               # Print messages
    mode='auto',             # 'min', 'max', or 'auto'
    min_delta=1e-4,          # Minimum improvement to reset patience
    cooldown=0,              # Epochs to wait before resuming normal operation
    min_lr=0                 # Lower bound for learning rate
)
```

### Typical Configuration

```python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce by half
    patience=5,        # Less patient than early stopping
    min_lr=1e-7,       # Stop reducing at 0.0000001
    verbose=1
)
```

## Common Keras Callbacks

### 1. TensorBoard

Real-time visualization during training:

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # Log weight histograms every epoch
    write_graph=True,        # Visualize model graph
    write_images=False,
    update_freq='epoch'
)

model.fit(X_train, y_train, epochs=50, callbacks=[tensorboard])

# Launch TensorBoard: tensorboard --logdir=logs/fit
```

### 2. CSVLogger

Log metrics to CSV for analysis:

```python
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False  # Overwrite file (True = append)
)

model.fit(X_train, y_train, epochs=50, callbacks=[csv_logger])

# Result: training_log.csv
# epoch,loss,accuracy,val_loss,val_accuracy
# 0,0.6931,0.5000,0.6920,0.5100
# 1,0.5234,0.7500,0.5100,0.7800
# ...
```

### 3. LearningRateScheduler

Custom learning rate schedule:

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    """Decrease learning rate by 10x every 10 epochs"""
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.1
    return lr

lr_schedule = LearningRateScheduler(scheduler, verbose=1)

model.fit(X_train, y_train, epochs=50, callbacks=[lr_schedule])

# Epoch 0: LR = 0.001
# Epoch 10: LR = 0.0001
# Epoch 20: LR = 0.00001
```

### 4. TerminateOnNaN

Stop training if loss becomes NaN:

```python
from tensorflow.keras.callbacks import TerminateOnNaN

terminate_nan = TerminateOnNaN()

model.fit(X_train, y_train, epochs=50, callbacks=[terminate_nan])

# If loss becomes NaN (e.g., exploding gradients), training stops immediately
```

### 5. Custom Callbacks

```python
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch"""
        if logs.get('val_accuracy') > 0.95:
            print(f"\nReached 95% accuracy at epoch {epoch}! Stopping training.")
            self.model.stop_training = True

custom_callback = CustomCallback()
model.fit(X_train, y_train, epochs=100, callbacks=[custom_callback])
```

## Production-Ready Callback Pipeline

```python
import os
import datetime
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
    TerminateOnNaN
)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 1. Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 2. Model Checkpoint
checkpoint = ModelCheckpoint(
    filepath='models/model_{epoch:02d}_{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 3. Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 4. TensorBoard
tensorboard = TensorBoard(
    log_dir=f'logs/fit/{timestamp}',
    histogram_freq=1,
    write_graph=True
)

# 5. CSV Logger
csv_logger = CSVLogger(f'logs/training_{timestamp}.csv')

# 6. Terminate on NaN
terminate_nan = TerminateOnNaN()

# Combine all callbacks
callbacks = [
    early_stop,
    checkpoint,
    reduce_lr,
    tensorboard,
    csv_logger,
    terminate_nan
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print(f"\nTraining complete!")
print(f"Best model saved in: models/")
print(f"Logs available in: logs/")
print(f"Launch TensorBoard: tensorboard --logdir=logs/fit")
```

## Callback Execution Order

Callbacks execute in the order they're defined:

```python
callbacks = [checkpoint, early_stop, reduce_lr]
# Execution at each epoch:
# 1. checkpoint evaluates → saves if best
# 2. early_stop evaluates → stops if patience exceeded
# 3. reduce_lr evaluates → reduces LR if plateau detected
```

**Order matters for dependent callbacks:**

```python
# CORRECT: Save before early stopping
callbacks = [ModelCheckpoint(...), EarlyStopping(...)]
# Model saved, then potentially stopped

# INCORRECT: Early stop before checkpoint
callbacks = [EarlyStopping(...), ModelCheckpoint(...)]
# If early stop triggers, checkpoint might not save last best model
```

## Monitoring Training Progress

### Accessing Callback History

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr]
)

# Access training metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()

# Check when early stopping triggered
stopped_epoch = len(history.history['loss'])
print(f"Training stopped at epoch: {stopped_epoch}")
```

## Best Practices

### 1. Always Use Validation Data

```python
# WRONG: No validation data, early stopping can't work
model.fit(X_train, y_train, epochs=100, callbacks=[early_stop])

# CORRECT: Provide validation data
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)
```

### 2. Set High `epochs` with Early Stopping

```python
# Let early stopping decide when to stop
model.fit(
    X_train, y_train,
    epochs=500,  # Set high, early stopping will interrupt
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)
```

### 3. Combine Complementary Callbacks

```python
# Early stopping + LR reduction + checkpointing
callbacks = [
    EarlyStopping(patience=15),
    ReduceLROnPlateau(patience=5),  # Trigger before early stopping
    ModelCheckpoint(save_best_only=True)
]
```

### 4. Log Everything

```python
callbacks = [
    CSVLogger('training.csv'),      # Metrics to CSV
    TensorBoard(log_dir='logs/'),   # Real-time visualization
    ModelCheckpoint('models/best.h5')  # Best model
]
```

## Key Takeaways

1. **EarlyStopping prevents overfitting** by monitoring validation metrics and stopping when improvement stops
2. **`patience` parameter** controls how many epochs to wait before stopping
3. **`restore_best_weights=True`** ensures model ends with best weights, not final weights
4. **ReduceLROnPlateau** automatically reduces learning rate when training plateaus
5. **Combine callbacks**: Early stopping + checkpointing + LR reduction for robust training
6. **TensorBoard** provides real-time visualization of training progress
7. **CSVLogger** records metrics for post-training analysis
8. **Callback order matters**: Define callbacks in logical order (checkpoint before early stop)
9. **Set high `epochs`** and let early stopping decide when to stop
10. **Always validate**: Early stopping requires validation data to monitor

## External Resources

- [Keras Callbacks API](https://keras.io/api/callbacks/) - Complete callback documentation
- [EarlyStopping Guide](https://keras.io/api/callbacks/early_stopping/) - Official EarlyStopping reference
- [Practical Tips for Training Deep Neural Networks](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/) - Learning rate strategies

