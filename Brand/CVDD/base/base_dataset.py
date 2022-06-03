from abc import ABC, abstractmethod
from torch.utils.data import Dataloader

class BaseADDataset(ABC):
    
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.n_classes = 2
        self.normal_classes = None # tuple with original class labels that define the normal class
        self.outlier_classes = None
        
        self.train_set = None
        self.test_set = None
    
    @abstractmethod
    def loaders(self, batch_size, shuffle, num_workers=0):
        pass
    
    def __repr__(self):
        return self.__class__.__name__