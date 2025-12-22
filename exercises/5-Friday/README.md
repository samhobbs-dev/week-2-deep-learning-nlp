# Friday Exercises: Transformers & Regularization

## Exercise Schedule

| Exercise | Type | Duration | Prerequisites |
|----------|------|----------|---------------|
| 01: Attention Basics | Implementation | 30 min | Demo 01 |
| 02: Build a Transformer | Implementation | 45 min | Demo 02 |
| 03: Add Regularization | Implementation | 30 min | Demo 03 |
| 04: Complete Model | Implementation | 45 min | Demo 04 |

**Total Time: ~2.5 hours**

---

## Exercise 01: Attention Basics (30 min)

### Goal
Use Keras MultiHeadAttention to process a sequence.

### Tasks

1. Open `starter_code/exercise_01_starter.py`
2. Complete the `TODO` sections:
   - Create a MultiHeadAttention layer with 4 heads
   - Apply it to the input sequence
   - Print the output shape

### Definition of Done
- [ ] Attention layer created
- [ ] Output shape is same as input shape
- [ ] Script runs without errors

---

## Exercise 02: Build a Transformer (45 min)

### Goal
Build a simple Transformer for text classification.

### Tasks

1. Open `starter_code/exercise_02_starter.py`
2. Complete the model:
   - Add embedding layer
   - Add position embeddings
   - Add MultiHeadAttention
   - Add classification head

### Definition of Done
- [ ] Model compiles successfully
- [ ] Model can predict on sample input
- [ ] Test accuracy > 50% (better than random)

---

## Exercise 03: Add Regularization (30 min)

### Goal
Add dropout and weight decay to prevent overfitting.

### Tasks

1. Open `starter_code/exercise_03_starter.py`
2. Add regularization:
   - Dropout(0.3) after Dense layers
   - L2 regularization to Dense layers
   - Early stopping callback

### Definition of Done
- [ ] Dropout layers added
- [ ] L2 regularizer applied
- [ ] Early stopping configured
- [ ] Gap between train/val accuracy reduced

---

## Exercise 04: Complete Model (45 min)

### Goal
Build a production-ready Transformer with all best practices.

### Tasks

1. Open `starter_code/exercise_04_starter.py`
2. Complete the production model with:
   - Proper embedding + position encoding
   - Attention with dropout
   - FFN with regularization
   - Model checkpointing

### Definition of Done
- [ ] Model trains successfully
- [ ] Best model saved to checkpoint
- [ ] Early stopping triggers
- [ ] Final test accuracy > 60%

---

## Solutions

Reference solutions are in `solutions/` folder.
Try to complete exercises before looking!
