from abc import ABC, abstractmethod
from .base_net import BaseNet
from .base_dataset import BaseADDataset


class BaseTrainer(ABC):
    
    def __init__(self, optimizer_name, lr, n_epochs, lr_milestones, batch_size,
                 weight_decay, device, n_jobs_dataloader):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        
        self.test_auc = None
        self.test_scores = None
    
    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        pass
    
    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        pass