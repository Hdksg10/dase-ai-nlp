import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from utils import load_datasets
from tasks import train_model, validation
from models.gru import GRUEncoderDecoder
from models.transformer import TransformerEncoderDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data hyperparameters
    parser.add_argument('--data_path', type=str, default='../data/', help="path to data")
    parser.add_argument('--valid_size', type=float, default=0.2, help="validation dataset ratio")
    parser.add_argument('--random_state', type=int, default=42, help="random state for splitting dataset")
    # model hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--vocab_size', type=int, default=1300, help="vocabulary size")
    # model utils
    parser.add_argument('--save_path', type=str, default='../ckpt/', help="save model to save_path with save_name")
    parser.add_argument('--save_name', type=str, default=None, help="save model to save_path with save_name")
    parser.add_argument('--save_model', default=False, action='store_true', help="save model to save_path")
    parser.add_argument('--load_path', type=str, default=None, help="load model from load_path")
    parser.add_argument('--load_model', default=False, action='store_true', help="load model from load_path")
    parser.add_argument('--trace_time', default=False, action='store_true', help="trace time for training")
    parser.add_argument('--test', default=False, action='store_true', help = 'evaluate on test set')
    parser.add_argument('--valid_loss', default=False, action='store_true', help="calculate validation loss every epoch")
    # model parameters
    parser.add_argument('--encoder_layers', type=int, default=2, help="number of layers in encoder")
    parser.add_argument('--decoder_layers', type=int, default=2, help="number of layers in decoder(should be the same as encoder)")
    parser.add_argument('--hidden_size', type=int, default=256, help="hidden size of both encoder and decoder, for GRU it is the number of hidden units, for Transformer it is the dimension feedforward network")
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4, help="number of heads in multi-head attention, in Transformer only")
    parser.add_argument('--direction', type=int, default=2, help="direction of encoder and decoder, 1 for unidirectional, 2 for bidirectional, in GRU only")
    parser.add_argument('model')
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader, test_loader = load_datasets(args.data_path, args.valid_size, args.random_state, args.batch_size)
    
    if args.model == 'GRU':
        model = GRUEncoderDecoder(vocab_size=args.vocab_size, 
                                  num_hiddens=args.hidden_size, 
                                  embedding_size=args.embedding_size, 
                                  encoder_layers=args.encoder_layers,
                                  decoder_layers=args.decoder_layers, 
                                  direction=args.direction)
    elif args.model == 'Transformer':
        model = TransformerEncoderDecoder(vocab_size=args.vocab_size, 
                                          num_hiddens=args.hidden_size, 
                                          embedding_size=args.embedding_size, 
                                          encoding_layers=args.encoder_layers,
                                          decoding_layers=args.decoder_layers, 
                                          num_heads=args.num_heads)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params}")
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_model(model, 
                criterion, 
                optimizer, 
                data_loader=train_loader, 
                device=device, 
                num_epochs=args.epochs, 
                save_path=args.save_path,
                save_model=args.save_model,
                save_name=args.save_name,
                load_path=args.load_path, 
                load_model=args.load_model,
                trace_time=args.trace_time,
                valid_loader=val_loader if args.valid_loss else None)
    if args.test:
        validation(model, test_loader, device)
    else:
        validation(model, val_loader, device)
    pass                                                                                                                                  