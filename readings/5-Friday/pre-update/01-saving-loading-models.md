# Saving and Loading Models

## Learning Objectives
- Understand different model serialization formats: SavedModel, H5, and weights-only
- Learn to save entire models vs. architecture + weights separately
- Master best practices for model persistence in production systems
- Apply model versioning and deployment strategies

## Why This Matters

Model persistence is critical for production machine learning. Training can take hours or days—losing that work would be catastrophic. Understanding model saving is essential because:

- **Production deployment**: Trained models must be saved, versioned, and deployed to serve predictions
- **Checkpoint recovery**: Long training runs need checkpoints to recover from failures
- **Collaboration**: Share trained models with team members or the research community
- **Transfer learning**: Save and reuse pre-trained models for downstream tasks
- **Model versioning**: Track model iterations for A/B testing and rollbacks

As you complete this week's deep learning and NLP fundamentals, model persistence bridges the gap between training and deployment. This is where your trained models become production assets.

## Model Serialization Formats

### 1. SavedModel (Recommended for Production)

**TensorFlow's native format**, optimized for serving and deployment.

```python
from tensorflow import keras

# Train model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Save in SavedModel format (entire directory)
model.save('saved_model/my_model')

# Directory structure:
# saved_model/my_model/
#   ├── assets/
#   ├── variables/
#   │   ├── variables.data-00000-of-00001
#   │   └── variables.index
#   └── saved_model.pb

# Load model
loaded_model = keras.models.load_model('saved_model/my_model')

# Use immediately
predictions = loaded_model.predict(X_test)
```

**Advantages:**
- **Production-ready**: Optimized for TensorFlow Serving
- **Complete**: Includes architecture, weights, optimizer state, and custom objects
- **Platform-independent**: Works across TensorFlow versions (with compatibility)
- **Recommended by TensorFlow**

**Disadvantages:**
- **Larger file size**: Stores more metadata
- **Directory structure**: Not a single file (can be zipped)

### 2. H5 Format (Legacy, but Common)

**HDF5 format**, a single-file binary format.

```python
# Save model to H5 file
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')
```

**Advantages:**
- **Single file**: Easier to distribute and version
- **Compact**: Smaller than SavedModel
- **Widely supported**: Works across TensorFlow/Keras versions

**Disadvantages:**
- **Legacy format**: TensorFlow recommends SavedModel for new projects
- **Less optimized**: Not as efficient for serving
- **Potential compatibility issues**: May break with major TensorFlow updates

### 3. Weights-Only Saving

Save only model weights, not architecture.

```python
# Save weights
model.save_weights('model_weights.h5')

# Load weights (must rebuild architecture first)
new_model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])
new_model.load_weights('model_weights.h5')

# Now model is ready (architecture + loaded weights)
```

**Advantages:**
- **Smallest file size**: Only parameters, no architecture metadata
- **Flexibility**: Can load weights into modified architectures (with care)

**Disadvantages:**
- **Requires manual architecture definition**: Must recreate exact same model
- **Error-prone**: Architecture mismatch causes silent failures or errors
- **No optimizer state**: Must recompile model

**When to use:**
- Transfer learning (load pre-trained weights into custom architecture)
- Fine-tuning (load base weights, modify top layers)
- Experimenting with architecture variations

## Saving Complete Models

### Best Practice: SavedModel

```python
import tensorflow as tf
from tensorflow import keras

# Build and train model
model = build_model()  # Your model architecture
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save complete model
model.save('models/sentiment_classifier')

# Load and use
loaded_model = keras.models.load_model('models/sentiment_classifier')
predictions = loaded_model.predict(X_test)
```

**What gets saved:**
- Model architecture
- Model weights
- Optimizer state (can resume training)
- Loss and metrics configuration
- Custom layers/objects (with registration)

### Saving to H5

```python
# Save to H5
model.save('sentiment_classifier.h5')

# Load from H5
loaded_model = keras.models.load_model('sentiment_classifier.h5')
```

## Saving Architecture and Weights Separately

### Save/Load Architecture (JSON)

```python
# Save architecture as JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Load architecture
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

from tensorflow.keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)

# Load weights separately
loaded_model.load_weights('model_weights.h5')

# Compile before use (optimizer state not saved)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Use Case: Architecture Exploration

```python
# Train baseline model
baseline_model = build_baseline()
baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
baseline_model.fit(X_train, y_train, epochs=10)

# Save baseline weights
baseline_model.save_weights('baseline_weights.h5')

# Experiment: Add more layers
experimental_model = build_experimental()  # Different architecture
# Can't load baseline weights (architecture mismatch)

# Better approach: Use SavedModel for versioning
baseline_model.save('models/baseline_v1')
experimental_model.save('models/experimental_v1')
```

## Custom Objects and Layers

### Saving Models with Custom Components

```python
# Custom layer
class CustomDenseLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    
    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({'units': self.units})
        return config

# Build model with custom layer
model = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    CustomDenseLayer(64),
    keras.layers.Dense(10, activation='softmax')
])

# Save model
model.save('custom_model.h5')

# Load model (must register custom layer)
loaded_model = keras.models.load_model(
    'custom_model.h5',
    custom_objects={'CustomDenseLayer': CustomDenseLayer}
)
```

**Without custom_objects**, you'll get an error:

```
ValueError: Unknown layer: CustomDenseLayer
```

### Custom Loss Functions

```python
# Custom loss
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred) + 0.1 * tf.abs(y_true - y_pred))

# Model with custom loss
model.compile(optimizer='adam', loss=custom_loss)
model.fit(X_train, y_train, epochs=10)

# Save
model.save('model_with_custom_loss.h5')

# Load (must provide custom loss)
loaded_model = keras.models.load_model(
    'model_with_custom_loss.h5',
    custom_objects={'custom_loss': custom_loss}
)
```

## Verifying Model Integrity After Loading

```python
# Save model
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Verify: Compare predictions
original_predictions = model.predict(X_test[:10])
loaded_predictions = loaded_model.predict(X_test[:10])

# Should be identical (within floating-point precision)
assert np.allclose(original_predictions, loaded_predictions, atol=1e-6)
print("Model integrity verified!")

# Verify: Compare weights
for layer_orig, layer_loaded in zip(model.layers, loaded_model.layers):
    weights_orig = layer_orig.get_weights()
    weights_loaded = layer_loaded.get_weights()
    for w_orig, w_loaded in zip(weights_orig, weights_loaded):
        assert np.allclose(w_orig, w_loaded)
print("Weights verified!")
```

## Model Versioning

### Semantic Versioning Strategy

```python
import os
from datetime import datetime

def save_versioned_model(model, base_path='models', version=None):
    """
    Save model with semantic versioning
    
    version format: v{major}.{minor}.{patch}
    Example: v1.0.0, v1.1.0, v2.0.0
    """
    if version is None:
        # Auto-generate version with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"v1_0_0_{timestamp}"
    
    model_path = os.path.join(base_path, f'model_{version}')
    model.save(model_path)
    print(f"Model saved: {model_path}")
    return model_path

# Usage
save_versioned_model(model, version='v1_0_0')
save_versioned_model(improved_model, version='v1_1_0')
save_versioned_model(major_update_model, version='v2_0_0')
```

### Metadata Logging

```python
import json

def save_model_with_metadata(model, model_name, metadata):
    """
    Save model with metadata for tracking
    
    metadata: dict with training info (accuracy, dataset, hyperparameters, etc.)
    """
    # Save model
    model_path = f'models/{model_name}'
    model.save(model_path)
    
    # Save metadata
    metadata_path = f'{model_path}/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model and metadata saved: {model_path}")

# Usage
metadata = {
    'version': '1.0.0',
    'training_date': '2025-12-09',
    'dataset': 'IMDB Reviews',
    'accuracy': 0.92,
    'loss': 0.21,
    'epochs': 10,
    'batch_size': 32,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'architecture': 'LSTM',
    'embedding_dim': 128,
    'lstm_units': 64
}

save_model_with_metadata(model, 'sentiment_v1', metadata)
```

## Best Practices

### 1. Save After Validation

```python
# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Only save if performance is acceptable
if test_accuracy > 0.85:
    model.save('models/production_model')
    print("Model saved for production")
else:
    print("Model performance below threshold, not saved")
```

### 2. Compress Large Models

```python
import shutil

# Save model
model.save('large_model')

# Compress to zip
shutil.make_archive('large_model', 'zip', 'large_model')

# File: large_model.zip (can be distributed easily)

# To load:
# 1. Extract zip
# 2. load_model('large_model')
```

### 3. Save Tokenizer with Model

```python
import pickle

# Train tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)

# Save model
model.save('models/text_classifier')

# Save tokenizer (required for inference!)
with open('models/text_classifier/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Production inference
loaded_model = keras.models.load_model('models/text_classifier')
with open('models/text_classifier/tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

# Preprocess new text
new_text = ["This movie was great"]
sequences = loaded_tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(sequences, maxlen=100)
prediction = loaded_model.predict(padded)
```

### 4. Environment Reproducibility

```python
# Save TensorFlow version info
import tensorflow as tf

metadata = {
    'tensorflow_version': tf.__version__,
    'keras_version': tf.keras.__version__,
    'python_version': '3.9.7',
    'model_path': 'models/sentiment_v1'
}

with open('models/sentiment_v1/environment.json', 'w') as f:
    json.dump(metadata, f, indent=4)
```

### 5. Test Loaded Model

```python
def test_model_loading(model_path, test_data):
    """Verify model loads correctly and produces expected output"""
    try:
        # Load model
        loaded_model = keras.models.load_model(model_path)
        
        # Test prediction
        predictions = loaded_model.predict(test_data)
        
        # Verify output shape
        assert predictions.shape[0] == test_data.shape[0]
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Prediction shape: {predictions.shape}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Usage
test_model_loading('models/sentiment_v1', X_test[:10])
```

## Production Deployment

### TensorFlow Serving (Docker)

```python
# Save model in SavedModel format
model.save('serving_models/sentiment_classifier/1')  # Version 1

# Directory structure:
# serving_models/
#   └── sentiment_classifier/
#       └── 1/
#           ├── assets/
#           ├── variables/
#           └── saved_model.pb
```

```bash
# Serve with TensorFlow Serving (Docker)
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/serving_models/sentiment_classifier,target=/models/sentiment_classifier \
  -e MODEL_NAME=sentiment_classifier \
  -t tensorflow/serving

# REST API endpoint: http://localhost:8501/v1/models/sentiment_classifier:predict
```

### Cloud Deployment (AWS SageMaker)

```python
# Save model for SageMaker
model.save('sagemaker_model/1')

# Package model with inference script
# See AWS SageMaker documentation for deployment
```

## Key Takeaways

1. **SavedModel format** (directory) is recommended for production deployment
2. **H5 format** (single file) is convenient for sharing and versioning
3. **Weights-only saving** is useful for transfer learning and architecture experiments
4. **Save entire models** (architecture + weights + optimizer) for reproducibility
5. **Custom objects** (layers, losses, metrics) require registration when loading
6. **Version models** semantically (v1.0.0, v1.1.0, v2.0.0) with metadata
7. **Save tokenizers and preprocessors** alongside models for production inference
8. **Verify model integrity** after loading by comparing predictions
9. **Compress large models** to zip for distribution
10. **Test loaded models** before deploying to production

## External Resources

- [TensorFlow Model Saving Guide](https://www.tensorflow.org/guide/keras/save_and_serialize) - Official documentation
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - Production model serving
- [Model Versioning Best Practices](https://ml-ops.org/content/model-versioning) - MLOps guidelines

