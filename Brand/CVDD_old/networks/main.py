from .network import CVDDNet
from .bert import BERT


def build_network(pretrained_model_name=None, attention_size=100, attention_heads=1):
    
    embedding = BERT(pretrained_model_name)    
    net = CVDDNet(embedding, attention_size, attention_heads)
    
    return net