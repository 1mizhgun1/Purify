import pandas as pd
from collections import defaultdict
import re
from Levenshtein import distance as levenshtein_distance
from functools import lru_cache

class SpellChecker:
    def __init__(self, dataset_path, max_distance=2):
        self.correct_words = set()
        self.error_to_correct = defaultdict(list)
        self.max_distance = max_distance
        
        df = pd.read_csv(dataset_path, sep=';', header=None, 
                        names=['correct', 'error', 'weight']).iloc[1:, :]
        
        for _, row in df.iterrows():
            correct = row['correct'].strip().lower()
            error = row['error'].strip().lower()
            weight = float(row['weight'])
            
            self.correct_words.add(correct)
            self.error_to_correct[error].append((correct, weight))
        
        for error in self.error_to_correct:
            self.error_to_correct[error].sort(key=lambda x: x[1], reverse=True)
        
        self.all_known_words = list(self.correct_words) + list(self.error_to_correct.keys())
    
    @lru_cache(maxsize=10000)
    def find_closest_word(self, word):
        if not word:
            return None
            
        word = word.lower()
        
        if word in self.correct_words:
            return word
        if word in self.error_to_correct:
            return self.error_to_correct[word][0][0]
        
        min_distance = float('inf')
        closest_word = None
        
        for known_word in self.all_known_words:
            current_distance = levenshtein_distance(word, known_word)
            if current_distance < min_distance and current_distance <= self.max_distance:
                min_distance = current_distance
                closest_word = known_word
        
        return closest_word
    
    def correct_spelling(self, word):
        word = word.lower().strip()
        
        if hasattr(self, '_spelling_cache') and word in self._spelling_cache:
            return self._spelling_cache[word]
        
        if word in self.correct_words:
            return word
        
        if word in self.error_to_correct:
            correction = self.error_to_correct[word][0][0]
            if not hasattr(self, '_spelling_cache'):
                self._spelling_cache = {}
            self._spelling_cache[word] = correction
            return correction
        
        closest_word = self.find_closest_word(word)
        if closest_word:
            if closest_word in self.error_to_correct:
                correction = self.error_to_correct[closest_word][0][0]
            else:
                correction = closest_word
            
            if not hasattr(self, '_spelling_cache'):
                self._spelling_cache = {}
            self._spelling_cache[word] = correction
            return correction
        
        return None
    
    def correct_text(self, text):
        tokens = re.findall(r"(\w+|\W+)", text)
        corrected_tokens = []
        for token in tokens:
            if token.strip() and token[0].isalpha():  
                correction = self.correct_spelling(token)
                if correction is not None:
                    if token[0].isupper():
                        correction = correction.capitalize()
                    corrected_tokens.append(correction)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        
        return ''.join(corrected_tokens)
