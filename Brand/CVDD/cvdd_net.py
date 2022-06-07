import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel


class BaseNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding = None
    
    def forward(self, *input):
        
        raise NotImplementedError
    
    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])


class BERT(nn.Module):
    
    def __init__(self, pretrained_model_name='bert-base-multilingual-uncased'):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.embedding = self.bert.embeddings
        self.embedding_size = self.embedding.word_embeddings.embedding_dim
        
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        self.bert.eval()
        x = self.bert(input_ids, attention_mask)
        hidden = x[0].transpose(0, 1)
        # hidden.shape = (sentence_length, batch_size, hidden_size)
        
        return hidden


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads

        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)

    def forward(self, hidden):
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        # Change to hidden.shape = (batch_size, sentence_length, hidden_size)
        hidden = hidden.transpose(0, 1)

        x = torch.tanh(self.W1(hidden))
        # x.shape = (batch_size, sentence_length, attention_size)

        x = torch.softmax(self.W2(x), dim=1)  # softmax over sentence_length
        # x.shape = (batch_size, sentence_length, n_attention_heads)

        A = x.transpose(1, 2)
        M = A @ hidden
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        return M, A


class CVDDNet(BaseNet):
    
    def __init__(self, pretrained_model_name, attention_size=100, attention_heads=1):
        super().__init__()
        
        self.pretrained_model = BERT(pretrained_model_name)
        self.hidden_size = self.pretrained_model.embedding_size
        
        # Set self-attention module
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size,
                                            attention_size=attention_size,
                                            n_attention_heads=attention_heads)
        
        self.c = nn.Parameter((torch.rand(1, attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        
        self.alpha = 0.0
    
    def forward(self, input_ids, attention_mask):
        
        # Need to fix here
        hidden = self.pretrained_model(input_ids, attention_mask)
        # hidden.shape = (sentence_length, batch_size, hidden_size)
        
        M, A = self.self_attention(hidden)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        cosine_dist = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = torch.softmax(-self.alpha * cosine_dist, dim=1)
        
        return cosine_dist, context_weights, A