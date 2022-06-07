from base.torchnlp_dataset import TorchnlpDataset
from torch.utils.data import Subset
from utils.text_encoders import TorchnlpBertTokenizer

import pandas as pd

class Invoice_Dataset(TorchnlpDataset):
    
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        
        self.n_classes = 2
        
        # load invoice dataset
        