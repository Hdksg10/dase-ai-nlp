import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError
    
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__()
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, x, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    bos = 1
    eos = 2
    pad = 0
    
    def __init__(self, encoder:Encoder, decoder:Decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder:Encoder = encoder
        self.decoder:Decoder = decoder

    def forward(self, x, y):
        encoded_state = self.encoder(x)
        decoder_state = self.decoder.init_state(encoded_state)
        decoded = self.decoder(y, decoder_state)
        return decoded 
    
        
    
    
    
