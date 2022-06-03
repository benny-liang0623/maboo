import torch
import torch.nn as nn

from transformers import BertModel

class CVDD:
    
    def __init__(self, model_name, attention_size=100, attention_heads=1):
        super().__init__()
        
        self.pretrained_model = BertModel.from_pretrained(model_name)
        self.hidden_size = self.pretrained_model.embedding_size
        
        # Set self-attention module
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.self_attention = nn.MultiheadAttention(self.attention_size, self.attention_heads)
        
        self.c = nn.Parameter((torch.rand(1, attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        
        self.alpha = 0.0
    
    def forward(self, x):
        
        # Need to fix here
        hidden = self.pretrained_model(x)
        M, A = self.self_attention(hidden)
        
        cosine_dist = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = torch.softmax(-self.alpha * cosine_dist, dim=1)
        
        return cosine_dist, context_weights, A