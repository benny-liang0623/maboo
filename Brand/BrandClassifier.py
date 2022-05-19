import torch.nn as nn

from transformers import BertModel

class ThresholdClassifier(nn.Module):
    def __init__(self, class_num, freeze=True, bert_model='bert-base-chinese'):
        super(ThresholdClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model)
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
        # distribution = torch.softmax(x)
        return x # distribution


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.02)
#         nn.init.zeros_(m.bias)