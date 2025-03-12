import nltk

# Download the correct data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Test tokenization
result = nltk.word_tokenize("This is a test sentence.")
print(f"Tokenized result: {result}")