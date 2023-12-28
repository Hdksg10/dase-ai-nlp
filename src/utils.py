import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

import pandas as pd


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
        # add eos token
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

def generate_length_mask(X:torch.Tensor, valid_length:torch.Tensor) -> torch.Tensor:
    batch_size = X.shape[0]
    max_length = X.shape[1]
    mask = torch.arange(max_length, device=X.device).expand(batch_size, max_length) >= valid_length.unsqueeze(1)
    return mask

def generate_square_subsequent_mask(sz) -> torch.Tensor:
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_masks(src:torch.Tensor, src_len:torch.Tensor, tgt:torch.Tensor, tgt_len:torch.Tensor, device):
    """Generate masks for transformer model from batch of sequences and their lengths
    Args:
        src (torch.Tensor): shape(batch_size, max_len) source sequence
        src_len (torch.Tensor): shape(batch_size,) source valid sequence
        tgt (torch.Tensor): shape(batch_size, max_len) target sequence
        tgt_len (torch.Tensor): shape(batch_size,) target valid sequence
        device (_type_): device to move masks to

    Returns:
        src_mask(torch.Tensor): shape(max_len, max_len) source mask
        tgt_mask(torch.Tensor): shape(max_len, max_len) target mask
        src_key_padding_mask(torch.Tensor): shape(batch_size, max_len) source key padding mask
        tgt_key_padding_mask(torch.Tensor): shape(batch_size, max_len) target key padding mask
        memory_key_padding_mask(torch.Tensor): shape(batch_size, max_len) memory key padding mask = src_key_padding_mask
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool) # encoder or decoder can use all infomation from src
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_key_padding_mask = generate_length_mask(src, src_len)
    tgt_key_padding_mask = generate_length_mask(tgt, tgt_len)
    memory_key_padding_mask = src_key_padding_mask
    # move masks to device
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device)
    tgt_key_padding_mask = tgt_key_padding_mask.to(device)
    memory_key_padding_mask = memory_key_padding_mask.to(device)
    return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    