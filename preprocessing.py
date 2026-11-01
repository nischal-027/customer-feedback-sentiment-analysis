
import re
import emoji
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')


# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))

# Preserve important negation words
negation_words = {'not', 'no', 'nor', 'never'}
stop_words = stop_words - negation_words

stemmer = PorterStemmer()


# Contractions dictionary
contractions = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "you're": "you are"
}


def clean_text(text):
    text = str(text)
    text = text.lower()

    for contraction, expansion in contractions.items():
        text = re.sub(
            r'\b' + re.escape(contraction) + r'\b',
            expansion,
            text
        )

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_review(text):

    # Step 1: Clean text
    text = clean_text(text)

    # Step 2: Tokenize
    tokens = word_tokenize(text)

    # Step 3: Remove stopwords
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Step 4: Apply stemming
    stemmed_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    # Step 5: Join tokens
    return ' '.join(stemmed_tokens)
