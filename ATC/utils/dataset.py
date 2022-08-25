import torch
from torch.utils.data import Dataset

class ATC_dataset(Dataset):
    def __init__(self, X, y, tokenizer, source_max_length, mode, device):
        self.encodings = tokenizer(X, padding='max_length', truncation=True, max_length=source_max_length)
        self.labels = y
        self.mode = mode
        self.device = device
        
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]).to(self.device) for key, val in self.encodings.items()}
        item['mode'] = self.mode
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[index]).to(self.device)
        return item