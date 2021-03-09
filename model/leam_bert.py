import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .leam import LEAM

class LEAMBert(LEAM):
    def __init__(self, bert, fine_tune_bert=True, **kwargs):
        super().__init__(**kwargs)
        self.V = bert
        self.V.requires_grad = fine_tune_bert

    def _embed_sentence(self, **x):
        return self.V(**x)[0]


