import sys
import torch
sys.path.append('..')

from utils.stop_words import ENGLISH_STOP_WORDS
from nltk import word_tokenize

class Tokenizer:
    def __init__(self, word_id_map, max_length=128, remove_stop_words=True):
        self.word_id_map = word_id_map
        self.inv_word_id_map = {v: k for k, v in word_id_map.items()}
        self.max_length = max_length
        self.stopwords = ENGLISH_STOP_WORDS
        self.remove_stop_words = remove_stop_words

    def preprocess(self, sentence):
        sentence = sentence.strip().lower()
        return sentence

    def encode(self, sentence, padding=True, truncation=True, return_tensors='pt'):
        sentence = self.preprocess(sentence)
        sentence = [self.word_id_map[c] for c in word_tokenize(sentence) if c not in self.stopwords]

        if len(sentence) < self.max_length:
            if padding is True:
                sentence = sentence + [0] * (self.max_length - len(sentence))
        else:
            if truncation is True:
                sentence = sentence[:self.max_length]

        if return_tensors == 'pt':
            sentence = torch.tensor(sentence)
        return sentence

    def decode(self, tokens):
        sentence = [self.inv_word_id_map[c] for c in tokens if c in self.inv_word_id_map]
        return sentence

    def batch_encode(self, sentences, padding=True, truncation=True):
        sentences = [self.encode(sentence, padding=padding, truncation=truncation) for sentence in sentences]
        return sentences
