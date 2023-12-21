import torch
import torch.optim as optim
import torch.nn as nn
import argparse

from utils import load_datasets
from tasks import train_model, validation
from models.gru import GRUEncoderDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--vocab_size', type=int, default=1300)
    parser.add_argument('--save_path', type=str, default='../ckpt/')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_path', type=str, default='../ckpt/model.pt')
    parser.add_argument('--load_model', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader, test_loader = load_datasets(args.data_path, args.test_size, args.random_state, args.batch_size)
    
    # a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(a.shape)
    # print(a)
    # a = a.unsqueeze(2)
    # print(a.shape)
    # print(a)
    
    # desc, diagn, desc_len, diagn_len = next(iter(train_loader))
    # desc = desc.to(torch.float32)
    # diagn = diagn.to(torch.float32)
    model = GRUEncoderDecoder(args.vocab_size, 256, 256, 2, direction=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, 
                criterion, 
                optimizer, 
                data_loader=train_loader, 
                device=device, 
                num_epochs=args.epochs, 
                save_path=args.save_path,
                save_model=args.save_model,
                load_path=args.load_path, 
                load_model=args.load_model)
    validation(model, val_loader, device)
    pass                                                                                                                                  