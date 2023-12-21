import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import MaskedSoftmaxCELoss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import tqdm

from models.base import EncoderDecoder
bos_token = 1
eos_token = 2
pad_token = 0

bos_pos_onehot = torch.zeros(1300, dtype=torch.long)
bos_pos_onehot[bos_token] = 1

eos_pos_onehot = torch.zeros(1300, dtype=torch.long)
eos_pos_onehot[eos_token] = 1

def train_model(model:nn.Module, criterion, optimizer:optim.Optimizer, data_loader:DataLoader, device, num_epochs, *args, **kwargs):
    model = model.to(device)
    model.train()
    # criterion = MaskedSoftmaxCELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (desc, diagn, desc_len, diagn_len) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            # Teacher forcing
            # Note: desc shape(batch_size, max_len)
            # Note: diagn shape(batch_size, max_len)
            # desc[:, desc_len] = eos_token
            y_in = diagn[:, :-1]
            y_in = torch.cat((torch.tensor(bos_token).repeat(diagn.shape[0], 1), y_in), dim=1)
            y_tgt = diagn[:, 1:]
            y_tgt = torch.cat((y_tgt, torch.tensor(pad_token).repeat(diagn.shape[0], 1)), dim=1)
            
            desc = desc.to(device)
            y_in = y_in.to(device)
            y_tgt = y_tgt.to(device)
            # print(y_tgt[0])
            # break

            optimizer.zero_grad()
            y_pred, _ = model(desc, y_in)
            y_pred = y_pred.permute(0, 2, 1) # shape(batch_size, vocab_size, max_len) for the K-dimensional case
            loss = criterion(y_pred, y_tgt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # break
        print(f'Epoch {epoch} loss: {running_loss / len(data_loader)}')
    pass

def predict(model: EncoderDecoder, src, max_len, device):
    """Generate prediction summary token by token, using greedy/beam search
    Use <bos> as first input token to decoder, and use generated token as next input token, until <eos> is generated or max length is reached.
    TODO: Beam search
    Args:
        model (_type_): _description_
        src (_type_): shape(1, seq_len)
        device (_type_): _description_
    """
    model.eval()
    src = src.to(device)
    encoded_out = model.encoder(src)
    state = model.decoder.init_state(encoded_out)
    # the decoder take input with shape(batch_size, seq_len)
    tgt = torch.tensor([bos_token], dtype=torch.long).unsqueeze(0).to(device)
    tgt_seq = []
    for _ in range(max_len):
        out, _ = model.decoder(tgt, state)
        out = out.argmax(dim=2)
        tgt = torch.cat((tgt, out[:, -1].unsqueeze(0)), dim=1)
        # print(f"step output: {tgt}")
        tgt_item = tgt[:, -1].unsqueeze(0).item()
        if tgt_item == eos_token:
            break
        tgt_seq.append(tgt_item)
    return tgt_seq

def bleu(hypothesis:list, reference:list):
    bleu_score = 0
    length = len(hypothesis)
    print(length)
    for i in range(length):
        hypo:str = ' '.join([str(num) for num in hypothesis[i]])
        ref:str = ' '.join([str(num) for num in reference[i]])
        bleu_score += sentence_bleu([ref], hypo, smoothing_function=SmoothingFunction().method1)
    bleu_score /= length
    return bleu_score
 
def rouge(hypothesis:list, reference:list):
    rouge_score = 0
    rouge = Rouge()
    length = len(hypothesis)
    for i in range(length):
        hypo:str = ' '.join([str(num) for num in hypothesis[i]])
        ref:str = ' '.join([str(num) for num in reference[i]])
        rouge_score += rouge.get_scores(hypo, ref)[0]['rouge-l']['f']
    rouge_score /= length
    
    return rouge_score

def validation(model, valid_loader, device, max_len = 130):
    model.eval()
    # desc, diagn, desc_len, diagn_len = next(iter(valid_loader))
    # pred = predict(model, desc[0], 130, device)
    # bleu_value = bleu(torch.tensor(pred), diagn[0][:diagn_len[0]])
    # print(bleu_value)
    hypothesis = []
    reference = []
    for i, (desc, diagn, desc_len, diagn_len) in tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader)):
        pred = predict(model, desc, max_len, device)
        diagn = diagn[:, 1:diagn_len[0]]
        reference.append(diagn.squeeze().tolist())
        hypothesis.append(pred)
    
    bleu_score = bleu(hypothesis, reference)
    rouge_score = rouge(hypothesis, reference)
    
    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE score: {rouge_score}")    