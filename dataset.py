from typing import List
import torch
from torch.utils.data import Dataset

class NewsQADataset(Dataset):
    """Dataset for simplified QA task on NewsQA"""

    def __init__(self, data: List[dict]):
        super().__init__()
    
    def __len__(self):
        pass    

    def __getitem__(self, idx):
        pass
