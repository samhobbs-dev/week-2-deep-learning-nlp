"""
Exercise 01: Tokenization Practice - Starter Code

Build a custom tokenizer with vocabulary management.

Prerequisites:
- Reading: 01-natural-language-processing-basics.md
- Reading: 02-tokenization-text-processing.md
- Demo: demo_01_tokenization_methods.py (REFERENCE FOR ALL PATTERNS)
"""

import re
from collections import Counter
import numpy as np

# ============================================================================
# TASK 1.1: Text Cleaning
# ============================================================================

class TextCleaner:
    """
    Text preprocessing utility.
    
    REGEX HINTS:
    - URL pattern: r'https?://\S+|www\.\S+'
    - Email pattern: r'\b[\w.-]+@[\w.-]+\.\w+\b'
    - Punctuation removal: r'[^\w\s]'
    
    SEE: demo_01_tokenization_methods.py lines 30-60 for cleaning examples
    """
    
    @staticmethod
    def lowercase(text):
        """Convert to lowercase"""
        return text.lower()
    
    @staticmethod
    def remove_urls(text):
        """
        Remove URLs (http://, https://, www.)
        Use: re.sub(pattern, '', text)
        """
        re.sub('http://', '', text)
        re.sub('https://', '', text)
        re.sub('www.', '', text)
        return text
    
    @staticmethod
    def remove_emails(text):
        """Remove email addresses"""
        re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '', text)
        return text
    
    @staticmethod
    def handle_punctuation(text, mode='remove'):
        """
        mode='remove': Delete all punctuation
        mode='separate': Add spaces around punctuation
        
        HINT for separate: re.sub(r'([^\w\s])', r' \1 ', text)
        """
        if mode == 'remove':
            re.sub(r'([^\w\s])', '', text)
        elif mode == 'separate':
            re.sub(r'([^\w\s])', r' \1 ', text)
        return text
    
    @staticmethod
    def normalize_whitespace(text):
        """
        Multiple spaces -> single space, strip ends.
        HINT: ' '.join(text.split())
        """
        text = ' '.join(text.split())
        return text
    
    @classmethod
    def clean(cls, text, remove_punct=True):
        """Apply all cleaning steps in order"""
        text = cls.lowercase(text)
        text = cls.remove_urls(text)
        text = cls.remove_emails(text)
        text = cls.handle_punctuation(text)
        text = cls.normalize_whitespace(text)
        return text


# ============================================================================
# TASK 1.2: Vocabulary Building
# ============================================================================

class Vocabulary:
    """
    Vocabulary manager for text encoding.
    
    SPECIAL TOKENS (use indices 0-3):
    - <PAD>: 0 (padding)
    - <UNK>: 1 (unknown word)
    - <START>: 2 (sentence start)
    - <END>: 3 (sentence end)
    """
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    
    def __init__(self, max_vocab_size=10000, min_freq=1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        self._built = False
    
    def _add_special_tokens(self):
        """Add special tokens at indices 0-3"""
        # YOUR CODE: Add PAD, UNK, START, END to word_to_idx and idx_to_word
        pass
    
    def build(self, texts):
        """
        Build vocabulary from list of texts.
        
        STEPS:
        1. Clean each text, split into words
        2. Count frequencies with Counter.update()
        3. Add special tokens first
        4. Add top N most common words (up to max_vocab_size)
        5. Only include words with freq >= min_freq
        """
        print("Building vocabulary...")
        
        # YOUR CODE HERE
        pass
    
    def get_index(self, word):
        """Return index for word, or UNK index if not found"""
        return self.word_to_idx.get(word, self.word_to_idx.get(self.UNK_TOKEN, 1))
    
    def get_word(self, idx):
        """Return word for index, or UNK if not found"""
        return self.idx_to_word.get(idx, self.UNK_TOKEN)
    
    @property
    def size(self):
        return len(self.word_to_idx)


# ============================================================================
# TASK 1.3: Tokenizer (Encoding/Decoding)
# ============================================================================

class Tokenizer:
    """
    Complete tokenizer: clean -> encode -> decode.
    
    SEE: demo_01_tokenization_methods.py lines 100-150 for encode/decode
    """
    
    def __init__(self, max_vocab_size=10000, max_length=100):
        self.vocab = Vocabulary(max_vocab_size=max_vocab_size)
        self.max_length = max_length
        self._fitted = False
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        self.vocab.build(texts)
        self._fitted = True
    
    def encode(self, text, add_special=False, pad=True):
        """
        Convert text to sequence of indices.
        
        STEPS:
        1. Clean text
        2. Split into words
        3. Convert each word to index using vocab.get_index()
        4. If add_special: prepend START, append END
        5. If pad: pad/truncate to max_length with PAD index
        
        Returns: list of integers
        """
        if not self._fitted:
            raise ValueError("Call fit() first")
        
        # YOUR CODE HERE
        return []
    
    def decode(self, indices, remove_special=True):
        """
        Convert indices back to text.
        
        STEPS:
        1. Convert each index to word using vocab.get_word()
        2. If remove_special: filter out PAD, START, END tokens
        3. Join with spaces
        """
        # YOUR CODE HERE
        return ''
    
    def encode_batch(self, texts, add_special=False):
        """Encode multiple texts, return numpy array"""
        encoded = [self.encode(t, add_special=add_special) for t in texts]
        return np.array(encoded)


# ============================================================================
# TESTING
# ============================================================================

def test_tokenizer():
    """Test your implementation"""
    corpus = [
        "I love natural language processing!",
        "NLP is a fascinating field of AI.",
        "Check out https://example.com for more info.",
        "Contact us at info@example.com today!"
    ]
    
    print("=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)
    
    # Test 1: Cleaning
    print("\n1. Text Cleaner:")
    test = "Check https://test.com and email@test.com for INFO!!!"
    # cleaned = TextCleaner.clean(test)
    # print(f"   Before: {test}")
    # print(f"   After:  {cleaned}")
    
    # Test 2: Vocabulary
    print("\n2. Vocabulary:")
    # tokenizer = Tokenizer(max_vocab_size=100, max_length=20)
    # tokenizer.fit(corpus)
    # print(f"   Vocab size: {tokenizer.vocab.size}")
    
    # Test 3: Encode/Decode
    print("\n3. Encode/Decode:")
    # encoded = tokenizer.encode("I love NLP", add_special=True)
    # decoded = tokenizer.decode(encoded)
    # print(f"   Encoded: {encoded}")
    # print(f"   Decoded: {decoded}")


if __name__ == "__main__":
    test_tokenizer()
