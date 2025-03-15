import re
import pymorphy3
import nltk
import pandas as pd

nltk.download('stopwords')
stopword_set = set(nltk.corpus.stopwords.words('russian'))
stopword_set_english = set(nltk.corpus.stopwords.words('english'))
# stopword_set = stopword_set.union({'это', 'который', 'весь', 'наш', 'свой', 'ещё', 'её', 'ваш', 'также', 'итак'})
stopword_set = stopword_set.union(stopword_set_english)

lemmatizer = pymorphy3.MorphAnalyzer()
lemmatizer_cache = {}

TOKEN_PATTERN = "[а-яёa-z]+"
def tokenize(text):
    return re.findall(TOKEN_PATTERN, text.lower())

def lemmatize(token):
    if lemmatizer.word_is_known(token):
        if token not in lemmatizer_cache:
            lemmatizer_cache[token] = lemmatizer.parse(token)[0].normal_form
        return lemmatizer_cache[token]
    return token

def preprocess_text(text):
    tokens = tokenize(text)
    lemmatized_tokens = [lemmatize(token) for token in tokens]
    cleared_tokens = [token for token in lemmatized_tokens if token not in stopword_set]
    return ' '.join(cleared_tokens)