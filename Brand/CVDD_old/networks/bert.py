import torch.nn as nn

from transformers import BertModel

class BERT(nn.Module):
    
    def __init__(self, pretrained_model_name='chinese-base-bert'):
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