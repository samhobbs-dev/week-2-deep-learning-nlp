"""Solution for Exercise 02"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

vocab_size = 10000
maxlen = 50
num_classes = 2

def build_transformer():
    inputs = keras.Input(shape=(maxlen,))
    
    # SOLUTION
    x = layers.Embedding(vocab_size, 64)(inputs)
    positions = tf.range(maxlen)
    x = x + layers.Embedding(maxlen, 64)(positions)
    
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attention_output)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

model = build_transformer()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

test_input = np.random.randint(1, vocab_size, size=(1, maxlen))
print(f"Prediction: {model.predict(test_input, verbose=0)}")
print("[OK] Complete!")
