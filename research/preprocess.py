"""Text preprocessing following paper 2026 (NLTK based).

Pipeline: tokenization -> stopword removal -> stemming -> lemmatization.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data() -> None:
    for resource in ('stopwords', 'punkt_tab', 'wordnet'):
        try:
            nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt_tab' \
                else nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_STOPWORDS = set(stopwords.words('english'))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_TOKEN_RE = re.compile(r'[^a-zA-Z\s]+')


def preprocess_text(text: str) -> str:
    tokens = _TOKEN_RE.sub(' ', text).lower().split()
    tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [_STEMMER.stem(t) for t in tokens]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)