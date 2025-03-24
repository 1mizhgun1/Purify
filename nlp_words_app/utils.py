from config import *
import re
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_lemma(word):
    return lemmatizer.parse(word)[0].normal_form

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = ' '.join(text.split())
    return text.lower()

def is_pronoun_or_stopword(word):
    return word in PRONOUNS or word in stopword_set

def split_compound_words(word):
    return re.findall(r'[А-Яа-яё]+', word)

word_cache = {}

def get_negative_words(text):
    cleaned_text = clean_text(text)
    # print(cleaned_text)
    negative_words = set()
    mat_words = list(re.findall(mat_regex, cleaned_text, re.VERBOSE))
    mat_words = set(split_compound_words("_".join(mat_words)))
    words = split_compound_words(text)
    # print(words)
    for word in words:
        if is_pronoun_or_stopword(word):
            continue

        if word in word_cache:
            if word_cache[word] == 'NGTV':
                negative_words.add(word)
            continue

        if word in mat_words:
            word_cache[word] = "NGTV"
            negative_words.add(word)
            continue

        if word == "сука":
            word_cache[word] = "NGTV"
            negative_words.add(word)
            continue

        lemma = get_lemma(word)
        
        if is_pronoun_or_stopword(lemma):
            continue
        
        if lemma in term_to_tag_dict:
            tag = term_to_tag_dict[lemma]
            word_cache[word] = tag
            if tag == 'NGTV':
                negative_words.add(word)
        else:
            lemma = checker.correct_spelling(word)
            print(lemma)
            if lemma in term_to_tag_dict:
                tag = term_to_tag_dict[lemma]
                word_cache[word] = tag
                if tag == 'NGTV':
                    negative_words.add(word)
            else:
                word_cache[word] = 'NEUT'

    print(negative_words)
    return list(negative_words)

