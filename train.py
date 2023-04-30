import argparse
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.tokenizer import tokenize


class YelpReviewDataset(Dataset):
    def __init__(self, df, vocab, train=True):
        self.df = df
        self.vocab = vocab
        self.seq_length = Config.TRAIN_SEQ_LENGTH if train else Config.TEST_SEQ_LENGTH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']  # text is a string
        token_list = tokenize(text)
        indices = torch.zeros(
            self.seq_length, dtype=torch.long)  # initialize as 0s
        for i, token in enumerate(token_list):
            if i >= self.seq_length:
                # Reached the maximum sequence length
                break
            if token in self.vocab:
                indices[i] = self.vocab[token]
            else:
                # Unknown token
                indices[i] = self.vocab['<unk>']

        label = self.df.loc[idx, 'label']
        return indices, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--model-choice', type=str,
                        required=True, choices=['lstm', 'transformer'])

    args = parser.parse_args()
    if args.model_choice == 'lstm':
        from config.lstm_config import LSTMConfig as Config
        from lstm.my_lstm import MyLSTM
    elif args.model_choice == 'transformer':
        from config.transformer_config import TransformerConfig as Config
        from transformer.my_transformer import MyTransformer

    # load vocab
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    # load data
    df = pd.read_csv(args.csv)
    # there are some rows with label = 'label', we need to remove them
    df = df[df['label'] != 'label']
    # convert label to int
    df['label'] = df['label'].astype(np.float64)

    train_data, _ = train_test_split(
        df, test_size=0.1, random_state=42)

    device = torch.device(
        'cuda' if Config.USE_GPU and torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(f'Device count: {torch.cuda.device_count()}')
        print(f'Current device index: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(0)}')
        print(
            f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')
        print()

    if Config.UPSAMPLE_NEGATIVE:
        # Upsample negative reviews according to Config.UPSAMPLE_RATIO
        train_data_pos = train_data[train_data['label'] == 1]
        train_data_neg = train_data[train_data['label'] == 0]
        train_data_neg_upsampled = train_data_neg.sample(
            n=int(len(train_data_neg) * Config.UPSAMPLE_RATIO), replace=True)
        train_data = pd.concat([train_data_pos, train_data_neg_upsampled])
        print(f"Upsampled negative reviews by {Config.UPSAMPLE_RATIO}x")

    # Reset dataframe index so that we can use df.loc[idx, 'text']
    train_data = train_data.reset_index(drop=True)
    print(
        f"Num positive reviews in training set: {len(train_data[train_data['label'] == 1])}")
    print(
        f"Num negative reviews in training set: {len(train_data[train_data['label'] == 0])}")

    train_dataset = YelpReviewDataset(train_data, vocab)

    # get dataloader from dataset
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # define model
    if args.model_choice == 'lstm':
        model = MyLSTM(vocab_size=len(vocab), embedding_size=Config.LSTM_EMBEDDING_SIZE,
                       hidden_size=Config.LSTM_HIDDEN_SIZE, num_layers=Config.LSTM_NUM_LAYERS,
                       dropout=Config.LSTM_DROUPOUT, num_classes=1, device=device)
    elif args.model_choice == 'transformer':
        model = MyTransformer(vocab_size=len(vocab), d_model=Config.D_MODEL,
                              ffn_hidden=Config.FFN_HIDDEN, output_dim=1, n_head=Config.N_HEAD,
                              drop_prob=Config.DROPOUT, max_len=Config.TRAIN_SEQ_LENGTH, 
                              device=device)
    # print num of parameters
    print(
        f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    model.train()
    model.to(device)
    # define binary cross entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # training loop
    print("Training...")
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.unsqueeze(1)  # (batch_size, 1)
            labels = labels.to(device)
            # forward
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm with loss value every 20 batches
            if (i+1) % 200 == 0:
                tqdm.write(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
                            Batch {i+1}/{len(train_loader)}, \
                            Batch Loss: {loss.item():.4f}, \
                            Average Loss: {total_loss / (i+1):.4f}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
              Average Loss: {total_loss / len(train_loader):.4f}")
        # save checkpoint
        torch.save(model.state_dict(),
                   f'models/checkpoints/{args.model_choice}_model_epoch{epoch+1}.pt')

    # save model
    torch.save(model.state_dict(), f'models/{args.model_choice}_model.pt')
    print(f"Model saved to models/{args.model_choice}_model.pt")
