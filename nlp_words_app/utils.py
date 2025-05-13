from config import *
import re
from functools import lru_cache
import redis
import pickle
import os
from logger_config import cache_logger
from dotenv import load_dotenv
from mistralai import Mistral
import json
import time
load_dotenv()

# Redis Settings
redis_url = os.environ['REDIS_URL']
redis_client = redis.Redis.from_url(redis_url)

# Misral Settings
mistral_api_key = os.environ["MISTRAL_API_KEY"]
mistral_client = Mistral(api_key=mistral_api_key)

# Задержка
API_DELAY_SECONDS = 2.0

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

def check_word_with_mistral(word: str) -> bool:
    """Проверка слова через Mistral API с повторными попытками"""
    max_retries = 3 
    base_delay = 2.0 
    
    for attempt in range(max_retries):
        try:
            time.sleep(API_DELAY_SECONDS)  
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": word}
            ]
            
            chat_response = mistral_client.chat.complete(
                model=model_mistral,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            response_json = json.loads(chat_response.choices[0].message.content)
            return bool(int(response_json.get("answer", 0)))
            
        except Exception as e:
            if "Service tier capacity exceeded" in str(e):
                if attempt < max_retries - 1:
                    current_delay = base_delay * (attempt + 1)  
                    cache_logger.warning(
                        f"Mistral API перегружен (попытка {attempt + 1}/{max_retries}). "
                        f"Повтор через {current_delay} сек..."
                    )
                    time.sleep(current_delay)
                    continue
                else:
                    cache_logger.error(
                        f"Достигнут лимит попыток для слова '{word}'. Ошибка: {str(e)}"
                    )
            else:
                cache_logger.error(f"Ошибка Mistral API для слова '{word}': {str(e)}")
            return False
    
    return False  

def is_material_word(word):
    """Проверка матерного слова с кэшированием и Mistral-подтверждением"""
    # Проверяем кэш Mistral
    mistral_cache_key = f"{MISTRAL_CACHE_PREFIX}{word}"
    cached_result = cache_get(mistral_cache_key)
    
    if cached_result is not None:
        return cached_result
    
    is_mat = bool(re.fullmatch(mat_regex, word, flags=re.VERBOSE))
    
    if is_mat:
        # is_confirmed = check_word_with_mistral(word)
        cache_set(mistral_cache_key, True, ttl=MISTRAL_CACHE_TTL)
        return True
    
    cache_set(mistral_cache_key, False, ttl=MISTRAL_CACHE_TTL)
    return False

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
    
    tag, _ = term_to_tag_dict.get(lemma, [None, 0.0])
    # if tag == 'NGTV' and prob < THRESHOLD:
        # tag = "NGTV"
    # else:
        # tag = "NEUT"
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
        if not re.search(r'[а-яёa-z]', word, flags=re.IGNORECASE):
            continue
        
        if len(word) < 2 or is_pronoun_or_stopword(word.lower()):
            continue
        
        word_upd = normalize_word(word).strip()

        # cache_logger.debug("")
        # cache_logger.debug(f"{word}: {word_upd}")

        lemma = get_lemma(word_upd)
        
        if is_pronoun_or_stopword(lemma):
            continue

        tag = get_word_tag(word_upd, lemma)
        if is_material_word(word_upd):
            cache_logger.debug(f"Lemma: {lemma}")
            cache_logger.debug(f"Tag: {tag}")
            if tag is None:
                cache_set(f"{WORD_TAG_PREFIX}word:{word}", "NGTV")
                cache_logger.debug(f"Mat: {is_material_word(word_upd)}")
                negative_words.add(word)
            else:
                cache_set(f"{WORD_TAG_PREFIX}word:{word}", "NEUT")
                cache_logger.debug(f"Neut: {word_upd}")

        # else:
        #     corrected_word = checker.correct_spelling(word_upd)
        #     if corrected_word:
        #         corrected_lemma = get_lemma(corrected_word)

        #         tag = get_word_tag(corrected_word, corrected_lemma)
        #         if tag == 'NGTV':
        #             # negative_words.add(word)
        #             continue
    
    print(negative_words)
    return list(negative_words)
