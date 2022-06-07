from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.trainer import CVDDTrainer

import json


class CVDD(object):
    
    def __init__(self, ad_score='context_dists_mean'):
        
        self.ad_score = ad_score
        
        self.net = None
        
        self.trainer = None
        self.optimizer_name = None
        
        self.train_dists = None
        
        self.test_dists = None
        self.test_att_weights = None
        
        self.results = {
            'context_vectors': None,
            'train_att_matrix': None,
            'test_att_matrix': None,
            'test_auc': None,
            'test_scores': None
        }
    
    def set_network(self, net_name, attention_size, attention_heads):
        self.net_name = net_name
        self.net = build_network(pretrain_model_name=net_name,
                                 attention_size=attention_size,
                                 attention_heads= attention_heads)
    
    def train(self, dataset: BaseADDataset, optimizer_name: str='adam', lr: float=0.001, n_epochs: int=150,
              lr_milestones: tuple=(), batch_size: int=64, lambda_p: float=1.0,
              alpha_scheduler: str='logarithmic', weight_decay: float=0.5e-6, device: str='cuda',
              n_jobs_dataloader: int=0):
        self.optimizer_name = optimizer_name
        self.trainer = CVDDTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, lambda_p, alpha_scheduler,
                                   weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)
        
        self.train_dists = self.trainer.train_dists
        self.results['context_vectors'] = self.trainer.c
        self.results['train_att_matrix'] = self.trainer.train_att_matrix
    
    def test(self, dataset: BaseADDataset, device: str='cuda', n_jobs_dataloader: int = 0):
        
        if self.trainer is None:
            self.trainer = CVDDTrainer(device, n_jobs_dataloader)
        
        self.trainer.test(dataset, self.net, ad_score=self.ad_score)
        
        self.test_dists = self.trainer.test_dists
        self.test_att_weights = self.trainer.test_att_weights
        self.results['test_att_matrix'] = self.trainer.test_att_matrix
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_scores'] = self.trainer.test_scores
    
    def save_model(self, export_path):
        pass
    
    def load_model(self, import_path, device: str='cuda'):
        pass
    
    def save_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.results, f)