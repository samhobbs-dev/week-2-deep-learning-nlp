"""Solution for Exercise 03"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import numpy as np

np.random.seed(42)
X = np.random.randn(500, 20).astype('float32')
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype('float32')
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

# SOLUTION
model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                   validation_data=(X_val, y_val),
                   epochs=50, 
                   callbacks=[early_stop],
                   verbose=1)
print("[OK] Complete!")
