import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base import Encoder
from models.base import Decoder
from utils import generate_masks, generate_length_mask, generate_square_subsequent_mask, PositionalEncoding
# from base import EncoderDecoder

class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, num_layers, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=0.1)
        self.layers = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=num_hiddens, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        
    def forward(self,
                src: torch.Tensor, 
                src_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            src (tensor): shape (batch_size, max_len)
            src_mask (tensor): shape (max_len, max_len) subsequent word mask to prevent attention to future words
            src_key_padding_mask (tensor): shape (batch_size, max_len) mask to prevent attention to padding tokens
        Returns:
            tensor: shape(batch_size, max_len, embedding_size)
        """
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x,
                                     mask = src_mask,
                                     src_key_padding_mask = src_key_padding_mask)
        return x

class TransformerDecoder(Decoder):
    def __init__(self, vocab_size, num_hiddens, embedding_size, num_layers, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=0.1)
        self.layers = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=num_hiddens, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)
    
    def init_state(self, enc_outputs, *args):
        return enc_outputs
        
    def forward(self, 
                tgt:torch.tensor, 
                memory:torch.tensor, 
                tgt_mask:torch.tensor,
                tgt_key_padding_mask:torch.tensor, 
                memory_key_padding_mask:torch.tensor) -> torch.Tensor:
        """_summary_

        Args:
            tgt (tensor): shape(batch_size, max_len)
            memory (tensor): shape(batch_size, max_len, embedding_size)
            tgt_mask (tensor): shape(max_len, max_len) subsequent word mask to prevent attention to future words
            tgt_key_padding_mask (tensor): shape(batch_size, max_len) mask to prevent attention to padding tokens
            memory_key_padding_mask (tensor): shape(batch_size, max_len) mask to prevent attention to padding tokens
            
        Returns:    
            tensor: shape(batch_size, max_len, vocab_size)
        """
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        x = self.transformer_decoder(x,
                                     memory,
                                     tgt_mask = tgt_mask,
                                     memory_mask = None,
                                     tgt_key_padding_mask = tgt_key_padding_mask,
                                     memory_key_padding_mask = memory_key_padding_mask)
        x = self.fc(x)
        return x
    
class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, embedding_size, encoding_layers, decoding_layers, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = TransformerEncoder(vocab_size, num_hiddens, embedding_size, encoding_layers, num_heads)
        self.decoder = TransformerDecoder(vocab_size, num_hiddens, embedding_size, decoding_layers, num_heads)
        # super().__init__(encoder, decoder)
        
        
    def forward(self, src:torch.Tensor, tgt:torch.Tensor, src_len:torch.Tensor, tgt_len:torch.Tensor):
        """_summary_

        Args:
            x (tensor): shape(batch_size, max_len)
            y_in (tensor): shape(batch_size, max_len)

        Returns:
            tensor: shape(batch_size, max_len, vocab_size)
        """
        device = src.device
        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = generate_masks(src, src_len, tgt, tgt_len, device)
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        dec_out = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return dec_out
        
    
    def predict(self, src, src_len, max_len, device):
        """Generate prediction summary token by token, using greedy/beam search
        Use <bos> as first input token to decoder, and use generated token as next input token, until <eos> is generated or max length is reached.
        """
        bos_token = 1
        eos_token = 2
        self.eval()
        src = src.to(device)
        src_len = src_len.to(device)
        src_mask = generate_square_subsequent_mask(src.shape[1])
        src_key_padding_mask = generate_length_mask(src, src_len)
        memory_key_padding_mask = src_key_padding_mask
        src_mask = src_mask.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)
        memory_key_padding_mask = memory_key_padding_mask.to(device)
        encoded_out = self.encoder(src, src_mask, src_key_padding_mask)
        state = self.decoder.init_state(encoded_out)
        # the decoder take input with shape(batch_size, seq_len)
        tgt = torch.tensor([bos_token], dtype=torch.long).unsqueeze(0).to(device)
        tgt_seq = []
        for i in range(max_len):
            length = i + 1
            length_tensor = torch.tensor([length]).to(device)
            tgt_mask = generate_square_subsequent_mask(length)
            tgt_mask = tgt_mask.to(device)
            tgt_key_padding_mask = generate_length_mask(tgt, length_tensor)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
            
            out = self.decoder(tgt, state, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
            out = out.argmax(dim=2)
            tgt = torch.cat((tgt, out[:, -1].unsqueeze(0)), dim=1)
            # print(f"step output: {tgt}")
            tgt_item = tgt[:, -1].unsqueeze(0).item()
            # print(tgt_item)
            if tgt_item == eos_token:
                break
            tgt_seq.append(tgt_item)
            
        return tgt_seq
        
    

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[1, 2, 3, 9], [4, 5, 6, 8]])
    x_pad_mask = torch.tensor([[True, True, True], [True, True, True]])
    y_pad_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    model = TransformerEncoderDecoder(
        vocab_size=10, 
        num_hiddens=256, 
        embedding_size=128, 
        encoding_layers=2, 
        decoding_layers=2, 
        num_heads=2
    )
    y_pred = model(x, y, x_pad_mask, y_pad_mask)
    print(y_pred.shape)
    pass