from config import *
import re
from functools import lru_cache
import redis
import pickle
import os
from logger_config import cache_logger
from dotenv import load_dotenv
load_dotenv()

redis_url = os.environ['REDIS_URL']
redis_client = redis.Redis.from_url(redis_url)

print("Cache Works?:")
try:
    response = redis_client.ping()
    print("True" if response else "False")
except redis.exceptions.ConnectionError:
    print("False")

LEMMA_PREFIX = "lemma:"
WORD_TAG_PREFIX = "word_tag:"
MAT_WORD_PREFIX = "mat_word:"

def cache_lemma(word: str, lemma: str):
    """Кэширование леммы"""
    redis_client.setex(
        f"{LEMMA_PREFIX}{word}",
        time=86400,
        value=pickle.dumps(lemma)
    )

def cache_get(key: str):
    """Универсальное получение из кэша"""
    cached = redis_client.get(key)
    if cached is None:
        return None
    try:
        value = pickle.loads(cached)
        return value
    except pickle.UnpicklingError:
        cache_logger.error(f"Cache ERROR: Could not unpickle value for key: {key}")
        return None

def cache_set(key: str, value, ttl: int = 3600):
    """Универсальное сохранение в кэш"""
    redis_client.setex(key, time=ttl, value=pickle.dumps(value))

def get_lemma(word):
    cache_key = f"{LEMMA_PREFIX}{word}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    
    lemma = lemmatizer.parse(word)[0].normal_form
    cache_set(cache_key, lemma, ttl=86400)
    return lemma

def is_material_word(word):
    """Проверка матерного слова с кэшированием"""
    cache_key = f"{MAT_WORD_PREFIX}{word}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    
    is_mat = bool(re.fullmatch(mat_regex, word, flags=re.VERBOSE))
    cache_set(cache_key, is_mat, ttl=3600)
    return is_mat

def get_word_tag(word, lemma):
    """Получение тега слова с кэшированием"""
    word_cache_key = f"{WORD_TAG_PREFIX}word:{word}"
    cached_tag = cache_get(word_cache_key)
    if cached_tag is not None:
        return cached_tag
    
    lemma_cache_key = f"{WORD_TAG_PREFIX}lemma:{lemma}"
    cached_tag = cache_get(lemma_cache_key)
    if cached_tag is not None:
        cache_set(word_cache_key, cached_tag)
        return cached_tag
    
    tag = term_to_tag_dict.get(lemma, 'NEUT')
    cache_set(word_cache_key, tag)
    cache_set(lemma_cache_key, tag)
    return tag

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = ' '.join(text.split())
    return text.lower()

def is_pronoun_or_stopword(word):
    return word in PRONOUNS or word in stopword_set

def split_compound_words(word):
    return re.findall(r'[А-Яа-яё]+', word)

def get_negative_words(text):
    cleaned_text = clean_text(text)
    negative_words = set()
    
    words = []
    for token in cleaned_text.split():
        words.extend(split_compound_words(token))
    
    for word in words:
        if len(word) < 2 or is_pronoun_or_stopword(word):
            continue

        if is_material_word(word) or word == "сука":
            cache_set(f"{WORD_TAG_PREFIX}word:{word}", "NGTV")
            negative_words.add(word)
            continue

        lemma = get_lemma(word)
        
        if is_pronoun_or_stopword(lemma):
            continue

        tag = get_word_tag(word, lemma)
        
        if tag == 'NGTV':
            negative_words.add(word)
        
        else:
            corrected_word = checker.correct_spelling(word)
            if corrected_word:
                corrected_lemma = get_lemma(corrected_word)
                tag = get_word_tag(corrected_word, corrected_lemma)
                if tag == 'NGTV':
                    negative_words.add(word)
    
    print(negative_words)
    return list(negative_words)
