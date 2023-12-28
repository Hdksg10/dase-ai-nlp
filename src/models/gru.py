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
        x = self.embdding(x)
        x, state = self.layers(x, state)
        x = self.fc(x)
        return x, state

class GRUEncoderDecoder(EncoderDecoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, encoder_layers, decoder_layers, direction = 1, *args, **kwargs):
        encoder = GRUEncoder(vocab_size, num_hiddens, embedding_size, encoder_layers, direction = direction)
        decoder = GRUDecoder(vocab_size, num_hiddens, embedding_size, decoder_layers, direction = direction)
        super().__init__(encoder, decoder)
    
    def forward(self, x, y):
        return super().forward(x, y)
    
    
    def predict(self, src, src_len, max_len, device):
        bos_token = 1
        eos_token = 2
        self.eval()
        src = src.to(device)
        encoded_out = self.encoder(src)
        state = self.decoder.init_state(encoded_out)
        # the decoder take input with shape(batch_size, seq_len)
        tgt = torch.tensor([bos_token], dtype=torch.long).unsqueeze(0).to(device)
        tgt_seq = []
        for _ in range(max_len):
            out, _ = self.decoder(tgt, state)
            out = out.argmax(dim=2)
            tgt = torch.cat((tgt, out[:, -1].unsqueeze(0)), dim=1)
            # print(f"step output: {tgt}")
            tgt_item = tgt[:, -1].unsqueeze(0).item()
            if tgt_item == eos_token:
                break
            tgt_seq.append(tgt_item)
        return tgt_seq
        pass
    
    
    
if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    model = GRUEncoderDecoder(10, 256, 128, 2, 2, 1)
    enc_out = model.encoder(x)
    state = model.decoder.init_state(enc_out)
    print(state.shape)
    y_pred, state = model.decoder(y, state)
    print(y_pred.shape)
    pass