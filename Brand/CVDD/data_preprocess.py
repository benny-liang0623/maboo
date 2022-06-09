import torch
from torch.utils.data import Dataset


class BrandDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, index):
        text = self.X.iloc[index]
        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_length,
            # pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'index':torch.tensor([index], dtype=torch.long),
            'ids':torch.tensor(ids, dtype=torch.long),
            'mask':torch.tensor(mask, dtype=torch.long),
            'target':torch.tensor(self.y.iloc[index], dtype=torch.float32)
        }
    
    def __len__(self):
        return len(self.X)


class CVDDDataset(Dataset):
    def __init__(self, X, tokenizer, max_length):
        self.X = X
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, index):
        text = self.X.values[index][0]
        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_length,
            # pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'index':torch.tensor([index], dtype=torch.long),
            'ids':torch.tensor(ids, dtype=torch.long),
            'mask':torch.tensor(mask, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.X)