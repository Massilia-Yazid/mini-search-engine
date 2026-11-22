import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    """
    Clean and normalize a text document
    """
    #convert text to lowercase
    text = text.lower()
    #remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    #split text into individual words
    tokens = text.split()
    #remove english stop words
    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]

    return tokens


