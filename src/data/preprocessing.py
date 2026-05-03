import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _stop_words]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str, stem: bool = False) -> str:
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def batch_preprocess(texts: List[str], stem: bool = False) -> List[str]:
    return [preprocess(t, stem=stem) for t in texts]
