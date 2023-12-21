import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MedicalDataset(Dataset):
    def __init__(self, data:pd.DataFrame, max_len=128):
        self.desc = data['description'].tolist()
        self.diagn = data['diagnosis'].tolist()
        self.max_len = max_len
        self.start_token = 1  
        self.end_token = 2    
        self.pad_token = 0   

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        description_list = [int(num) for num in self.desc[idx].split()]
        diagnosis_list = [int(num) for num in self.diagn[idx].split()]
        desc_len = len(description_list)
        diagn_len = len(diagnosis_list)
        description = description_list[:self.max_len]
        diagnosis = diagnosis_list[:self.max_len]
        description = torch.tensor(description)
        diagnosis = torch.tensor(diagnosis)

        description = torch.cat((description, torch.tensor([self.pad_token] * (self.max_len - len(description)), dtype=torch.long)))
        diagnosis = torch.cat((diagnosis, torch.tensor([self.pad_token] * (self.max_len - len(diagnosis)), dtype=torch.long)))
        if desc_len >= self.max_len:
            desc_len = self.max_len-1
        if diagn_len >= self.max_len:
            diagn_len = self.max_len-1
        description[desc_len] = 2
        diagnosis[diagn_len] = 2
        return description, diagnosis, desc_len, diagn_len

def load_datasets(data_path, test_size=0.2, random_state=42, batch_size=64):
    train_data = pd.read_csv(data_path + 'train.csv')
    test_data = pd.read_csv(data_path + 'test.csv')
    train_data, val_data = train_test_split(train_data, test_size=test_size, random_state=random_state)
    
    train_dataset = MedicalDataset(train_data)
    val_dataset = MedicalDataset(val_data)
    test_dataset = MedicalDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    return train_loader, val_loader, test_loader 
    pass

if __name__ == '__main__':
    """check vocab size 1300
    """
    data_path = "../data/"
    train_data = pd.read_csv(data_path + 'train.csv')
    test_data = pd.read_csv(data_path + 'test.csv')
    max_char = 0
    for id, row in test_data.iterrows():
        desc = row['description'].split()
        diagn = row['diagnosis'].split()
        desc_int = [int(num) for num in desc]
        diagn_int = [int(num) for num in diagn]
        
        min_desc = min(desc_int)
        min_diagn = min(diagn_int)
        max_desc = max(desc_int)
        max_diagn = max(diagn_int)
        if max_desc > max_char:
            max_char = max_desc
        if max_diagn > max_char:
            max_char = max_diagn
        if min_desc < 3:
            print(id, 'desc', desc_int)

        if min_diagn < 3:
            print(id, 'diagn', diagn_int)
    for id, row in train_data.iterrows():
        desc = row['description'].split()
        diagn = row['diagnosis'].split()
        desc_int = [int(num) for num in desc]
        diagn_int = [int(num) for num in diagn]
        
        min_desc = min(desc_int)
        min_diagn = min(diagn_int)
        max_desc = max(desc_int)
        max_diagn = max(diagn_int)
        if max_desc > max_char:
            max_char = max_desc
        if max_diagn > max_char:
            max_char = max_diagn
        if min_desc < 3:
            print(id, 'desc', desc_int)

        if min_diagn < 3:
            print(id, 'diagn', diagn_int)
    pass
    print(max_char)

def mask_seq(seq, seq_len):
    mask = torch.zeros(seq.shape, dtype=torch.float32)
    for i in range(seq.shape[0]):
        mask[i, :seq_len[i]] = 1
    return mask

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss