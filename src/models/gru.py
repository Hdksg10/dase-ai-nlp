import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base import Encoder
from models.base import Decoder
from models.base import EncoderDecoder

class GRUEncoder(Encoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, num_layers, direction = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction
        self.bidirectional = False
        if direction == 2:
            self.bidirectional = True
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.layers = nn.GRU(embedding_size, num_hiddens, num_layers, batch_first=True, bidirectional=self.bidirectional)
    
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): shape (batch_size, max_len)

        Returns:
            out, h_n: out shape(max_len, batch_size, num_hiddens), h_n shape (num_layers, num_hiddens) or (num_layers, batch_size, num_hiddens) containing the final hidden state
        """
        x = self.embedding(x) # shape(batch_size, max_len, embedding_size)
        return self.layers(x)
    

class GRUDecoder(Decoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, num_layers, direction = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction
        self.bidirectional = False
        if direction == 2:
            self.bidirectional = True
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embdding = nn.Embedding(vocab_size, embedding_size)
        self.layers = nn.GRU(embedding_size, num_hiddens, num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(direction * num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, *args):
        """

        Args:
            enc_outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = enc_outputs[1]
        return state
    
    def forward(self, x, state):
        """_summary_

        Args:
            x : shape(batch_size, max_len)
            state : shape(num_layers, max_len, num_hiddens)

        Returns:
            x: shape(batch_size, max_len, vocab_size)
        """
        # x = x.permute(1, 0, 2) # shape(max_len, batch_size, vocab_size)
        # print(state.shape)
        # state = state[:, -1, :]
        # state = state.unsqueeze(1)
        # state = state.repeat
        # print(state.shape)
        x = self.embdding(x)
        x, state = self.layers(x, state)
        # print(x.shape)
        x = self.fc(x)
        return x, state

class GRUEncoderDecoder(EncoderDecoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, num_layers, direction = 1, *args, **kwargs):
        encoder = GRUEncoder(vocab_size, num_hiddens, embedding_size, num_layers, direction = 1)
        decoder = GRUDecoder(vocab_size, num_hiddens, embedding_size, num_layers, direction = 1)
        super().__init__(encoder, decoder)
    
    def forward(self, x, y):
        return super().forward(x, y)
    
    # def train_step(self, x, y, criterion, optimizer):
    #     optimizer.zero_grad()
    #     y_pred = self.forward(x)
    #     loss = criterion(y_pred, y)
    #     loss.backward()
    #     optimizer.step()
    #     return loss.item()
    
    def predict(self, x):
        
        pass
    
    
    
if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    model = GRUEncoderDecoder(10, 256, 128, 2)
    enc_out = model.encoder(x)
    state = model.decoder.init_state(enc_out)
    print(state.shape)
    y_pred, state = model.decoder(y, state)
    print(y_pred.shape)
    pass