import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import tqdm
import time

from models.base import EncoderDecoder
from models.transformer import TransformerEncoderDecoder
bos_token = 1
eos_token = 2
pad_token = 0

# bos_pos_onehot = torch.zeros(1300, dtype=torch.long)
# bos_pos_onehot[bos_token] = 1

# eos_pos_onehot = torch.zeros(1300, dtype=torch.long)
# eos_pos_onehot[eos_token] = 1

def load(model:nn.Module, load_path) -> nn.Module:
    if load_path:
        model.load_state_dict(torch.load(load_path))
    print(f'Loaded model from {load_path}')
def save(model:nn.Module, save_path, name=None):
    if not name:
        name = model.__class__.__name__ + '.pt'
    if save_path:
        torch.save(model.state_dict(), save_path + name)
    pass
    
def valid_loss(model, valid_loader, device, criterion):
    model.eval()
    loss = 0
    for i, (desc, diagn, desc_len, diagn_len) in tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader)):
        y_in = diagn[:, :-1]
        y_in = torch.cat((torch.tensor(bos_token).repeat(diagn.shape[0], 1), y_in), dim=1)
        y_tgt = diagn[:, 1:]
        y_tgt = torch.cat((y_tgt, torch.tensor(pad_token).repeat(diagn.shape[0], 1)), dim=1)
        
        desc = desc.to(device)
        y_in = y_in.to(device)
        y_tgt = y_tgt.to(device)
        desc_len = desc_len.to(device)
        diagn_len = diagn_len.to(device)
        if isinstance(model, EncoderDecoder):
            y_pred, _ = model(desc, y_in)
        elif isinstance(model, TransformerEncoderDecoder):
            y_pred = model(desc, y_in, desc_len, diagn_len)
        y_pred = y_pred.permute(0, 2, 1) # change to shape(batch_size, vocab_size, max_len) for the K-dimensional case
        loss += criterion(y_pred, y_tgt).item()
    return loss / len(valid_loader)

def train_model(model:nn.Module, criterion, optimizer:optim.Optimizer, data_loader:DataLoader, device, num_epochs, *args, **kwargs):
    load_path = kwargs.get('load_path', None)
    load_model = kwargs.get('load_model', False)
    trace_time = kwargs.get('trace_time', False)
    save_model = kwargs.get('save_model', False)
    save_path = kwargs.get('save_path', None)
    save_name = kwargs.get('save_name', None)
    valid_loader = kwargs.get('valid_loader', None)
    print(load_model)
    model = model.to(device)
    if load_model is True:
        print(f'Loading model from {load_path}')
        load(model, load_path)
        return
    # criterion = MaskedSoftmaxCELoss()
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (desc, diagn, desc_len, diagn_len) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            # Teacher forcing
            
            y_in = diagn[:, :-1]
            y_in = torch.cat((torch.tensor(bos_token).repeat(diagn.shape[0], 1), y_in), dim=1)
            y_tgt = diagn[:, 1:]
            y_tgt = torch.cat((y_tgt, torch.tensor(pad_token).repeat(diagn.shape[0], 1)), dim=1)
            
            desc = desc.to(device)
            y_in = y_in.to(device)
            y_tgt = y_tgt.to(device)
            desc_len = desc_len.to(device)
            diagn_len = diagn_len.to(device)
            optimizer.zero_grad()
            if isinstance(model, EncoderDecoder):
                y_pred, _ = model(desc, y_in)
            elif isinstance(model, TransformerEncoderDecoder):
                y_pred = model(desc, y_in, desc_len, diagn_len)
            y_pred = y_pred.permute(0, 2, 1) # change to shape(batch_size, vocab_size, max_len) for the K-dimensional case
            loss = criterion(y_pred, y_tgt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / y_tgt.shape[0]
            if epoch == 10 and i == 10:
                print(y_pred.argmax(dim=1))
                print(y_tgt)
        # validation loss
        if valid_loader:
            loss = valid_loss(model, valid_loader, device, criterion)
            print(f'Epoch {epoch} loss: {running_loss / len(data_loader)}, validation loss: {loss}')
        else:
            print(f'Epoch {epoch} loss: {running_loss / len(data_loader)}')
    end_time = time.time()
    if save_model:
        save(model, save_path, save_name)
    if trace_time:
        print(f'Training time: {end_time - start_time}')
    pass

def predict(model: EncoderDecoder, src, src_len, max_len, device):
    """Generate prediction summary token by token, using greedy/beam search
    Use <bos> as first input token to decoder, and use generated token as next input token, until <eos> is generated or max length is reached.
    TODO: Beam search
    Args:
        model (_type_): _description_
        src (_type_): shape(1, seq_len)
        device (_type_): _description_
    """

    return model.predict(src, src_len, max_len, device)

def bleu(hypothesis:list, reference:list):
    bleu_score = 0
    length = len(hypothesis)
    # print(reference)
    for i in range(length):
        if hasattr(reference[i], '__len__'):
            ref:str = ' '.join([str(num) for num in reference[i]])
        else:
            ref:str = str(reference[i])
        hypo:str = ' '.join([str(num) for num in hypothesis[i]])
        bleu_score += sentence_bleu([ref], hypo, smoothing_function=SmoothingFunction().method1)
    bleu_score /= length
    return bleu_score
 
def rouge(hypothesis:list, reference:list):
    rouge_score = 0
    rouge = Rouge()
    length = len(hypothesis)
    for i in range(length):
        if hasattr(reference[i], '__len__'):
            ref:str = ' '.join([str(num) for num in reference[i]])
        else:
            ref:str = str(reference[i])
        hypo:str = ' '.join([str(num) for num in hypothesis[i]])
        rouge_score += rouge.get_scores(hypo, ref)[0]['rouge-1']['r']
    rouge_score /= length
    
    return rouge_score

def validation(model, valid_loader, device, max_len = 128):
    model.eval()
    hypothesis = []
    reference = []
    for i, (desc, diagn, desc_len, diagn_len) in tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader)):
        pred = predict(model, desc, desc_len, max_len, device)
        diagn = diagn[:, 1:diagn_len[0]]
        reference.append(diagn.squeeze().tolist())
        hypothesis.append(pred)
        # if i == 10:
        #     break
    
    bleu_score = bleu(hypothesis, reference)
    rouge_score = rouge(hypothesis, reference)
    
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE score: {rouge_score}")    