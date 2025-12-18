"""
Demo 01: Simple RNN for Sequence Prediction

This demo shows trainees how to:
1. Understand RNN architecture and hidden states
2. Build a simple RNN for text classification
3. Visualize how RNNs process sequences step-by-step
4. Recognize RNN limitations (vanishing gradients)

Learning Objectives:
- Understand recurrent connections and hidden state
- Build RNN models in Keras
- Visualize sequence processing

References:
- Written Content: 01-recurrent-neural-networks.md
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: RNN Conceptual Understanding
# ============================================================================

print("=" * 70)
print("PART 1: Understanding Recurrent Neural Networks")
print("=" * 70)

print("\nRNN Key Concept: Memory Through Hidden State")
print("-" * 50)
print("""
Unlike feedforward networks, RNNs maintain a HIDDEN STATE
that carries information from previous time steps.

For a sentence: "I love NLP"

Feedforward: Processes each word independently
  "I"    -> prediction
  "love" -> prediction (no memory of "I")
  "NLP"  -> prediction (no memory of "I" or "love")

RNN: Processes sequentially with memory
  "I"    -> h1 (hidden state 1)
  "love" + h1 -> h2 (remembers "I")
  "NLP"  + h2 -> h3 (remembers "I love")
  h3 -> final prediction
""")

# ============================================================================
# PART 2: RNN Forward Pass Visualization
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Visualizing RNN Forward Pass")
print("=" * 70)

# Manual RNN implementation for understanding
class SimpleRNNCell:
    """Simple RNN cell for educational purposes"""
    
    def __init__(self, input_dim, hidden_dim):
        # Initialize weights randomly
        np.random.seed(42)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x, h_prev):
        """Single RNN step"""
        # h_t = tanh(W_xh * x + W_hh * h_prev + b)
        h_new = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        return h_new

# Create simple RNN
input_dim = 4   # Simple 4-dimensional input
hidden_dim = 3  # 3-dimensional hidden state

rnn = SimpleRNNCell(input_dim, hidden_dim)

# Sequence of 5 time steps
sequence = np.random.randn(5, input_dim)

print(f"\nInput sequence shape: {sequence.shape} (5 time steps, 4 features)")
print(f"Hidden state dimension: {hidden_dim}")

# Process sequence step by step
h = np.zeros(hidden_dim)  # Initial hidden state
hidden_states = [h]

print(f"\nInitial hidden state h0: {h}")
print("\nProcessing sequence:")
print("-" * 50)

for t, x_t in enumerate(sequence):
    h = rnn.forward(x_t, h)
    hidden_states.append(h)
    print(f"t={t}: x_t shape={x_t.shape}, h_{t+1}={h.round(3)}")

print(f"\nFinal hidden state h5: {h.round(3)}")
print("This final state encodes information from ALL time steps!")

# Visualize hidden states over time
hidden_states = np.array(hidden_states)

plt.figure(figsize=(12, 4))
for i in range(hidden_dim):
    plt.plot(hidden_states[:, i], marker='o', label=f'Hidden unit {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Hidden State Value')
plt.title('RNN Hidden State Evolution Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(6), ['h0', 'h1', 'h2', 'h3', 'h4', 'h5'])
plt.tight_layout()
plt.savefig('rnn_hidden_states.png', dpi=150)
print("\n Hidden state visualization saved")

# ============================================================================
# PART 3: Building RNN in Keras
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Building RNN in Keras")
print("=" * 70)

# Create sample data (sentiment analysis)
texts = [
    "I love this movie",
    "This film is terrible",
    "Amazing performance",
    "Worst movie ever",
    "Highly recommend"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Tokenize


tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10, padding='pre')
y = np.array(labels)

print(f"\nTokenized sequences:")
for text, seq in zip(texts, X):
    print(f"  '{text}' -> {seq}")

# Build RNN model
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    layers.SimpleRNN(16, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
], name='simple_rnn_classifier')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel architecture:")
model.summary()

# Train
print("\nTraining RNN...")
history = model.fit(X, y, epochs=50, verbose=0)

# Predict
predictions = model.predict(X, verbose=0)
print("\nPredictions:")
for text, pred in zip(texts, predictions):
    sentiment = "positive" if pred > 0.5 else "negative"
    print(f"  '{text}' -> {sentiment} ({pred[0]:.2f})")

# ============================================================================
# PART 4: Return Sequences vs Final State
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: return_sequences=True vs False")
print("=" * 70)

print("\nreturn_sequences=False (default):")
print("  Returns only the FINAL hidden state")
print("  Output shape: (batch_size, hidden_units)")
print("  Use for: Classification (many-to-one)")

print("\nreturn_sequences=True:")
print("  Returns hidden state at EVERY time step")
print("  Output shape: (batch_size, sequence_length, hidden_units)")
print("  Use for: Sequence-to-sequence, stacking RNNs")

# Demonstrate
sample_input = np.random.randn(1, 10, 32)  # 1 sample, 10 time steps, 32 features

# Without return_sequences
rnn_no_seq = layers.SimpleRNN(16, return_sequences=False)
output_no_seq = rnn_no_seq(sample_input)
print(f"\nreturn_sequences=False: input {sample_input.shape} -> output {output_no_seq.shape}")

# With return_sequences
rnn_with_seq = layers.SimpleRNN(16, return_sequences=True)
output_with_seq = rnn_with_seq(sample_input)
print(f"return_sequences=True:  input {sample_input.shape} -> output {output_with_seq.shape}")

# ============================================================================
# PART 5: Stacking RNNs
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Stacking Multiple RNN Layers")
print("=" * 70)

stacked_model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    
    # First RNN layer - return sequences for next RNN
    layers.SimpleRNN(32, return_sequences=True),
    
    # Second RNN layer - return final state
    layers.SimpleRNN(16),
    
    layers.Dense(1, activation='sigmoid')
], name='stacked_rnn')

print("\nStacked RNN architecture:")
stacked_model.summary()

print("\n Why stack RNNs?")
print("- First layer: learns low-level patterns")
print("- Second layer: learns high-level abstractions")
print("- Similar to stacking Dense layers")

# ============================================================================
# PART 6: Bidirectional RNN
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Bidirectional RNN")
print("=" * 70)

print("\nBidirectional RNN processes sequence BOTH ways:")
print("  Forward:  'I' -> 'love' -> 'NLP'")
print("  Backward: 'NLP' -> 'love' -> 'I'")
print("  Final: Concatenate both hidden states")

bidirectional_model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    layers.Bidirectional(layers.SimpleRNN(16)),
    layers.Dense(1, activation='sigmoid')
], name='bidirectional_rnn')

print("\nBidirectional RNN architecture:")
bidirectional_model.summary()

print("\n When to use Bidirectional:")
print("+ Text classification (can see whole sentence)")
print("+ Named Entity Recognition")
print("- NOT for text generation (can't see future during generation)")

# ============================================================================
# PART 7: Vanishing Gradient Problem
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: The Vanishing Gradient Problem")
print("=" * 70)

print("""
RNNs struggle with LONG sequences because of vanishing gradients.

For a 100-word sentence:
  Gradient at word 1 depends on gradients from words 2-100
  Each backprop step multiplies by tanh derivative (<= 1)
  Gradient shrinks: 0.9^100 = 0.00003 (vanishes!)

Result:
  - RNNs can't learn long-term dependencies
  - Effectively only "remembers" last ~10 words
  - Solution: LSTM (next demo!)

Practical limit: SimpleRNN works for ~10-20 time steps
""")

# Demonstrate with long sequence
print("\nDemonstrating: Training RNN on long sequences...")

# Create long sequence data
long_sequence_length = 50
X_long = np.random.randint(0, 1000, (100, long_sequence_length))
y_long = np.random.randint(0, 2, 100)

long_rnn = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=long_sequence_length),
    layers.SimpleRNN(32),
    layers.Dense(1, activation='sigmoid')
])

long_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f"Training SimpleRNN on sequence length {long_sequence_length}...")
history_long = long_rnn.fit(X_long, y_long, epochs=20, verbose=0)

print(f"Final accuracy: {history_long.history['accuracy'][-1]:.3f}")
print("\n RNNs struggle with long sequences. LSTM solves this!")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Simple RNN")
print("=" * 70)

print("\nKey Takeaways:")
print("1. RNNs maintain hidden state across time steps")
print("2. Hidden state = 'memory' of previous inputs")
print("3. return_sequences=True for stacking RNNs")
print("4. Bidirectional RNNs see context from both directions")
print("5. Vanishing gradients limit RNNs to short sequences")
print("6. LSTM (next) solves the vanishing gradient problem")

print("\n" + "=" * 70)

