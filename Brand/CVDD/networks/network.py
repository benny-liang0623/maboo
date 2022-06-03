import torch
import torch.nn as nn

from base.base_net import BaseNet
from networks.self_attention import SelfAttention
class CVDDNet(BaseNet):
    
    def __init__(self, pretrained_model, attention_size=100, attention_heads=1):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.embedding_size
        
        # Set self-attention module
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size,
                                            attention_size=attention_size,
                                            attention_heads=attention_heads)
        
        self.c = nn.Parameter((torch.rand(1, attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        
        self.alpha = 0.0
    
    def forward(self, x):
        
        # Need to fix here
        hidden = self.pretrained_model(x)
        # hidden.shape = (sentence_length, batch_size, hidden_size)
        
        M, A = self.self_attention(hidden)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        cosine_dist = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = torch.softmax(-self.alpha * cosine_dist, dim=1)
        
        return cosine_dist, context_weights, A