from transformers import BertTokenizer
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX

import torch

class TorchnlpBertTokenizer(BertTokenizer):
    
    def __init__(self, vocab_file, do_lowercase=False, append_eos=False):
        super().__init__(vocab_file, do_lowercase=do_lowercase)
        self.append_eos = append_eos
        
        self.itos = list(self.vocab.keys())
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        
        self.vocab = self.itos
        self.vocab_size = len(self.vocab)
        
    def encode(self, text, eos_index=DEFAULT_EOS_INDEX, unknown_index=DEFAULT_UNKNOWN_INDEX):
        text = self.tokenize(text)
        unknown_index = self.stoi['[UNK]']
        vector = [self.stoi.get(token, unknown_index) for token in text]
        if self.append_eos:
            vector.append(eos_index)
        return torch.LongTensor(vector)
    
    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)