import re

import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stopword = stopwords.words('english')


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_number(text: str) -> str:
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'', text)


def remove_punctuation(text: str) -> str:
    # remove punctuations
    punctuations = '@#!?+&*[]—-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, ' ')
    return text


def remove_blank_space(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    text = ' '.join([word for word in text.split() if word not in stopword])
    return text


def lemmatize_lower(text: str) -> str:
    # lemmatize the word
    stop_words = [word for word in text.split() if word not in stopword]
    text = ' '.join(
        [lemmatizer.lemmatize(contractions.fix(to_lowercase(text))) for txt in text.split() if txt not in stop_words])
    return text


def clean_text(text: str) -> str:
    text = remove_number(text)
    text = remove_punctuation(text)
    text = remove_blank_space(text)
    text = lemmatize_lower(text)
    text = remove_stopwords(text)
    return text
