import torch.nn as nn

from transformers import BertModel, XLMRobertaModel


class BaselineClassifier(nn.Module):
    '''
    baseline classifier:
        train acc: 43.89%
        valid acc: 47.43%
    '''
    def __init__(self, class_num, freeze=True, bert_model='bert-base-chinese'):
        super(BaselineClassifier, self).__init__()
        self.bert_model = XLMRobertaModel.from_pretrained(bert_model)
        self.classifier = nn.Sequential(
            nn.Linear(768, class_num),
            # nn.Tanh(),
            # nn.Linear(512, 256),
            # nn.Tanh(),
            # nn.Linear(256, 121)    
        )
        
        if freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        x = self.bert_model(input_ids, attention_mask)
        # print(x[0].shape) (batch, max_length, 768)
        # print(x[1].shape) (batch, 768)
        x = self.classifier(x[1])
        return x


class DeepClassifier(nn.Module):
    '''
    more advanced classifier:
        train acc:
        valid acc: 
    '''
    def __init__(self, class_num, freeze=True, bert_model='bert-base-chinese'):
        super(DeepClassifier, self).__init__()
        self.bert_model = XLMRobertaModel.from_pretrained(bert_model)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.Tanh(),
            nn.Linear(1024, class_num)    
        )
        
        if freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        x = self.bert_model(input_ids, attention_mask)
        # print(x[0].shape) (batch, max_length, 768)
        # print(x[1].shape) (batch, 768)
        x = self.classifier(x[1])
        return x