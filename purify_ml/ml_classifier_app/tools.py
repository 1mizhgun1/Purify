import re
import pymorphy3
import nltk
import numpy as np
from gensim.models import Word2Vec

from config import W2V_WEIGHTS

nltk.download('stopwords')
stopword_set = set(nltk.corpus.stopwords.words('russian'))
stopword_set_english = set(nltk.corpus.stopwords.words('english'))
# stopword_set = stopword_set.union({'это', 'который', 'весь', 'наш', 'свой', 'ещё', 'её', 'ваш', 'также', 'итак'})
stopword_set = stopword_set.union(stopword_set_english)

lemmatizer = pymorphy3.MorphAnalyzer()
lemmatizer_cache = {}

model_w2v = Word2Vec.load(W2V_WEIGHTS)
print("Модель загружена")
# similar_words = model_w2v.wv.most_similar('оскорбление', topn=5) 
# print(similar_words)

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

def preprocess_for_w2v(text):
    s = preprocess_text(text)
    s = s.split()
    input_ = [s]
    X_vector_inference = np.array([np.mean([model_w2v.wv[word] for word in sentence if word in model_w2v.wv] or [np.zeros(768)], axis=0) for sentence in input_])
    return X_vector_inference