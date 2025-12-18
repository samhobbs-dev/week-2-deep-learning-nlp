"""
Demo 02: LSTM Comparison with SimpleRNN

This demo shows trainees how to:
1. Understand LSTM architecture (gates, cell state)
2. Compare LSTM vs SimpleRNN performance
3. Visualize how LSTMs handle long sequences
4. Build LSTM models in Keras

Learning Objectives:
- Understand LSTM gates (forget, input, output)
- See the performance difference on long sequences
- Build production-ready LSTM models

References:
- Written Content: 02-lstm-long-short-term-memory.md
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os

# ============================================================================
# PART 1: LSTM Architecture Explanation
# ============================================================================

print("=" * 70)
print("PART 1: LSTM Architecture")
print("=" * 70)

print("""
LSTM: Long Short-Term Memory

Problem with SimpleRNN:
  - Vanishing gradients: Can't learn long-term dependencies
  - Information decays over time steps

LSTM Solution: Three gates + Cell State

1. FORGET GATE: What to remove from memory
   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
   "Should I forget the previous subject when I see a new one?"

2. INPUT GATE: What new info to add
   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
   C_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)
   "Is this word important enough to remember?"

3. CELL STATE UPDATE: The memory highway
   C_t = f_t * C_{t-1} + i_t * C_tilde
   "Forget some old info, add some new info"

4. OUTPUT GATE: What to output
   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   "What's relevant for the current prediction?"

Key Insight:
  - Cell state (C_t) uses ADDITION, not multiplication
  - Gradients flow backward without vanishing!
  - Can remember information for 100+ time steps
""")

# ============================================================================
# PART 2: Building LSTM in Keras
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Building LSTM in Keras")
print("=" * 70)

# Load IMDB dataset
print("Loading IMDB sentiment dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Sequence length: {max_length}")
print(f"Vocabulary size: 10,000")

# Build SimpleRNN model
def build_simple_rnn():
    return keras.Sequential([
        layers.Embedding(10000, 64, input_length=max_length),
        layers.SimpleRNN(32),
        layers.Dense(1, activation='sigmoid')
    ], name='SimpleRNN')

# Build LSTM model
def build_lstm():
    return keras.Sequential([
        layers.Embedding(10000, 64, input_length=max_length),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ], name='LSTM')

# Compare architectures
rnn_model = build_simple_rnn()
lstm_model = build_lstm()

# Build models (required for count_params in Keras 3)
rnn_model.build(input_shape=(None, max_length))
lstm_model.build(input_shape=(None, max_length))

print("\nSimpleRNN Model:")
print(f"  Parameters: {rnn_model.count_params():,}")

print("\nLSTM Model:")
print(f"  Parameters: {lstm_model.count_params():,}")
print(f"  (LSTM has ~4x more parameters due to 4 gates)")

# ============================================================================
# PART 3: Training Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Training Comparison (SimpleRNN vs LSTM)")
print("=" * 70)

# Use subset for faster demo
x_train_sub = x_train[:5000]
y_train_sub = y_train[:5000]
x_test_sub = x_test[:1000]
y_test_sub = y_test[:1000]

# Train SimpleRNN
print("\nTraining SimpleRNN...")
rnn_model = build_simple_rnn()
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TensorBoard for SimpleRNN
log_dir_rnn = "logs/rnn_comparison/simple_rnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_rnn, exist_ok=True)
tb_rnn = keras.callbacks.TensorBoard(log_dir=log_dir_rnn, histogram_freq=1)

start_time = time.time()
history_rnn = rnn_model.fit(
    x_train_sub, y_train_sub,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tb_rnn],
    verbose=1
)
rnn_time = time.time() - start_time

rnn_test_loss, rnn_test_acc = rnn_model.evaluate(x_test_sub, y_test_sub, verbose=0)
print(f"SimpleRNN Test Accuracy: {rnn_test_acc:.4f} (Time: {rnn_time:.1f}s)")

# Train LSTM
print("\nTraining LSTM...")
lstm_model = build_lstm()
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TensorBoard for LSTM
log_dir_lstm = "logs/rnn_comparison/lstm_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_lstm, exist_ok=True)
tb_lstm = keras.callbacks.TensorBoard(log_dir=log_dir_lstm, histogram_freq=1)

start_time = time.time()
history_lstm = lstm_model.fit(
    x_train_sub, y_train_sub,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tb_lstm],
    verbose=1
)
lstm_time = time.time() - start_time

lstm_test_loss, lstm_test_acc = lstm_model.evaluate(x_test_sub, y_test_sub, verbose=0)
print(f"LSTM Test Accuracy: {lstm_test_acc:.4f} (Time: {lstm_time:.1f}s)")

# ============================================================================
# PART 4: View Training Comparison (TensorBoard)
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Visualizing Training Comparison")
print("=" * 70)

print("\nTraining metrics are now logged to TensorBoard!")
print(f"To compare SimpleRNN vs LSTM, run:")
print(f"  tensorboard --logdir=logs/rnn_comparison")
print(f"  Then navigate to http://localhost:6006")
print("\nIn TensorBoard:")
print("  - Use the 'Runs' selector to toggle between simple_rnn_* and lstm_*")
print("  - Compare loss and accuracy curves side by side")
print("  - Notice LSTM's superior performance on long sequences")

# Print summary
print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"SimpleRNN: {rnn_test_acc:.4f} accuracy, {rnn_time:.1f}s training")
print(f"LSTM:      {lstm_test_acc:.4f} accuracy, {lstm_time:.1f}s training")
print(f"\nLSTM wins by {(lstm_test_acc - rnn_test_acc)*100:.1f}% accuracy!")

# ============================================================================
# PART 5: LSTM Variants
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: LSTM Variants and Configurations")
print("=" * 70)

# Stacked LSTM
stacked_lstm = keras.Sequential([
    layers.Embedding(10000, 64, input_length=max_length),
    layers.LSTM(64, return_sequences=True),  # Return sequences for stacking
    layers.LSTM(32),                          # Final LSTM
    layers.Dense(1, activation='sigmoid')
], name='Stacked_LSTM')

stacked_lstm.build(input_shape=(None, max_length))
print("\nStacked LSTM:")
print(f"  Parameters: {stacked_lstm.count_params():,}")

# Bidirectional LSTM
bidirectional_lstm = keras.Sequential([
    layers.Embedding(10000, 64, input_length=max_length),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
], name='Bidirectional_LSTM')

bidirectional_lstm.build(input_shape=(None, max_length))
print("\nBidirectional LSTM:")
print(f"  Parameters: {bidirectional_lstm.count_params():,}")
print("  (2x parameters: forward + backward)")

# LSTM with dropout
lstm_dropout = keras.Sequential([
    layers.Embedding(10000, 64, input_length=max_length),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
], name='LSTM_with_Dropout')

print("\nLSTM with Dropout:")
print("  dropout=0.2: Drop 20% of input connections")
print("  recurrent_dropout=0.2: Drop 20% of recurrent connections")

# ============================================================================
# PART 6: GRU Alternative
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: GRU - Simpler Alternative to LSTM")
print("=" * 70)

print("""
GRU (Gated Recurrent Unit):
- Simplified LSTM with 2 gates instead of 3
- Fewer parameters, faster training
- Often comparable performance

LSTM: forget, input, output gates (3 gates)
GRU:  reset, update gates (2 gates)
""")

gru_model = keras.Sequential([
    layers.Embedding(10000, 64, input_length=max_length),
    layers.GRU(32),
    layers.Dense(1, activation='sigmoid')
], name='GRU')

gru_model.build(input_shape=(None, max_length))
print(f"GRU Parameters: {gru_model.count_params():,}")
print(f"LSTM Parameters: {lstm_model.count_params():,}")
print(f"\nGRU has ~75% of LSTM parameters (3 gates vs 4)")

# Train GRU for comparison
print("\nTraining GRU...")
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_gru = gru_model.fit(
    x_train_sub, y_train_sub,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=0
)
gru_time = time.time() - start_time

gru_test_loss, gru_test_acc = gru_model.evaluate(x_test_sub, y_test_sub, verbose=0)
print(f"GRU Test Accuracy: {gru_test_acc:.4f} (Time: {gru_time:.1f}s)")

# Final comparison
print("\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)
print(f"SimpleRNN: {rnn_test_acc:.4f} accuracy")
print(f"LSTM:      {lstm_test_acc:.4f} accuracy")
print(f"GRU:       {gru_test_acc:.4f} accuracy")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: LSTM Comparison")
print("=" * 70)

print("\nKey Takeaways:")
print("1. LSTM uses gates to control information flow")
print("2. Cell state acts as 'memory highway' - gradients don't vanish")
print("3. LSTM significantly outperforms SimpleRNN on long sequences")
print("4. LSTM has ~4x more parameters than SimpleRNN")
print("5. GRU is a simpler alternative with similar performance")
print("6. Use dropout for regularization (dropout + recurrent_dropout)")

print("\nWhen to use what:")
print("- SimpleRNN: Very short sequences (< 20 tokens)")
print("- LSTM: Long sequences, complex dependencies")
print("- GRU: When LSTM is too slow, need fewer parameters")
print("- Bidirectional: Classification tasks (see both directions)")

print("\n" + "=" * 70)

