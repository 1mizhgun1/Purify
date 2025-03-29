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
    
    tag, prob = term_to_tag_dict.get(lemma, ["NEUT", 0.0])
    if tag == 'NGTV' and prob < THRESHOLD:
        tag = "NGTV"
    else:
        tag = "NEUT"
    cache_set(word_cache_key, tag)
    cache_set(lemma_cache_key, tag)
    return tag

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = ' '.join(text.split())
    return text

def is_pronoun_or_stopword(word):
    return word in PRONOUNS or word in stopword_set

def normalize_word(word):
    word_tmp = word
    word_tmp = word_tmp.replace('bI', 'ы')
    word_tmp = word_tmp.replace('3.14', 'пи')
    word_tmp = word_tmp.replace('3,14', 'пи')
    
    for replacement_group in replace_map:
        for pattern, replacement in replacement_group.items():
            word_tmp = word_tmp.replace(pattern, replacement)
    
    word_tmp = re.sub(r'^[^а-яёa-z]+|[^а-яёa-z]+$', '', word_tmp, flags=re.IGNORECASE)
    word_tmp = re.sub(r'([аеёиоуыэюяaeiouy])\1+', r'\1', word_tmp, flags=re.IGNORECASE)
    word_tmp = re.sub(r'([а-яёa-z])[^а-яёa-z]+([а-яёa-z])', r'\1-\2', word_tmp, flags=re.IGNORECASE)
    word_tmp = re.sub(r'-+', '-', word_tmp)
    return word_tmp.lower()

def get_negative_words(text):
    cleaned_text = clean_text(text)

    negative_words = set()
    
    for word in cleaned_text.split():
        if len(word) < 2 or is_pronoun_or_stopword(word.lower()):
            continue
        
        word_upd = normalize_word(word).strip()

        cache_logger.debug("")
        cache_logger.debug(f"{word}: {word_upd}")

        lemma = get_lemma(word_upd)
        
        if is_pronoun_or_stopword(lemma):
            continue

        tag = get_word_tag(word_upd, lemma)
        cache_logger.debug(f"Lemma: {lemma}")
        cache_logger.debug(f"Tag: {tag}")
        cache_logger.debug(f"Mat: {is_material_word(word_upd)}")
        cache_logger.debug("")
        
        if tag == 'NGTV':
            negative_words.add(word)
            continue

        elif is_material_word(word_upd):
            cache_set(f"{WORD_TAG_PREFIX}word:{word}", "NGTV")
            negative_words.add(word)
            continue
        
        else:
            corrected_word = checker.correct_spelling(word_upd)
            if corrected_word:
                corrected_lemma = get_lemma(corrected_word)

                tag = get_word_tag(corrected_word, corrected_lemma)
                if tag == 'NGTV':
                    negative_words.add(word)
                    continue
    
    print(negative_words)
    return list(negative_words)
