import argparse
import importlib
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.tokenizer import MyTokenizer


class YelpReviewDataset(Dataset):
    def __init__(self, df, vocab, max_seq_length):
        self.df = df
        self.vocab = vocab
        self.seq_length = max_seq_length
        self.tokenizer = MyTokenizer(
            vocab, max_seq_length, remove_stopwords=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']  # text is a string
        # get indices of tokens from the text
        indices = torch.tensor(self.tokenizer(text), dtype=torch.long)
        label = self.df.loc[idx, 'label']
        return indices, label


def train(model, Config, criterion, optimizer, device, checkpoints, train_loader, val_loader=None):
    print("Training...")
    train_losses, val_losses, val_accuracy = [], [], []
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        model.train()
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.unsqueeze(1)  # (batch_size, 1)
            labels = labels.to(device)

            # Apply label smoothing by changing labels from 0, 1 to 0.1, 0.9
            if Config.LABEL_SMOOTHING:
                labels = (1 - Config.LABEL_SMOOTHING_EPSILON) * labels + \
                    Config.LABEL_SMOOTHING_EPSILON * (1 - labels)

            # forward
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            if Config.GRADIENT_CLIP:
                # clip gradient norm
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=Config.GRADIENT_CLIP_VALUE)
            optimizer.step()

            # update tqdm with loss value every 20 batches
            if (i+1) % 200 == 0:
                tqdm.write(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
                            Batch {i+1}/{len(train_loader)}, \
                            Batch Loss: {loss.item():.4f}, \
                            Average Loss: {total_loss / (i+1):.4f}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, \
              Average Loss: {total_loss / len(train_loader):.4f}")
        # save loss for plot
        train_losses.append(total_loss / len(train_loader))
        # save checkpoint
        if checkpoints:
            try:
                checkpoint_path = f'models/checkpoints/{args.model_choice}_model_epoch{epoch+1}.pt'
                torch.save(model.state_dict(), checkpoint_path)
            except OSError as e:
                print(f"Could not save checkpoint at epoch {epoch+1}, error: {e}")

        # evaluate on validation set if necessary
        if args.val_in_training:
            model.eval()
            with torch.no_grad():
                total_loss = total = TP = TN = 0
                print(f"Validation at epoch {epoch + 1}...")
                for data, labels in tqdm(val_loader):
                    data = data.to(device)
                    labels = labels.unsqueeze(1).to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    predicted = torch.round(torch.sigmoid(outputs))
                    total += labels.size(0)

                    TP += ((predicted == 1) & (labels == 1)).sum().item()
                    TN += ((predicted == 0) & (labels == 0)).sum().item()
                print(f"Validation Accuracy: {(TP + TN) / total:.4f}")
                print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
                val_losses.append(total_loss / len(val_loader))
                val_accuracy.append((TP + TN) / total)

    if args.plot_loss:
        # plot loss
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Val loss')
        plt.xticks(np.arange(len(train_losses)),
                   np.arange(1, len(train_losses)+1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{args.model_choice} Loss')
        plt.legend(frameon=False)
        plt.savefig(f'{args.model_choice}_loss.png')
        # clear plot for next plot
        plt.clf()
        print(f"Loss plot saved to {args.model_choice}_loss.png")

        # plot accuracy
        plt.plot(val_accuracy, label='Val accuracy')
        plt.xticks(np.arange(len(train_losses)),
                   np.arange(1, len(train_losses)+1))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{args.model_choice} Accuracy')
        plt.legend(frameon=False)
        plt.savefig(f'{args.model_choice}_accuracy.png')
        print(f"Accuracy plot saved to {args.model_choice}_accuracy.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--model-choice', type=str,
                        required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--load-trained', action='store_true', default=False)
    parser.add_argument('--val-in-training', action='store_true',
                        default=False, help='Do a validation run after every epoch')
    parser.add_argument('--plot-loss', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoints', action='store_true', default=False)
    args = parser.parse_args()

    if args.plot_loss:
        if not args.val_in_training:
            raise ValueError("Cannot plot loss without validation in training. "
                             "Please set --val-in-training to True")
        import matplotlib.pyplot as plt

    if args.model_choice == 'lstm':
        Config = importlib.import_module('config.' + args.config).LSTMConfig
        from lstm.my_lstm import MyLSTM
    elif args.model_choice == 'transformer':
        Config = importlib.import_module('config.' + args.config).TransformerConfig
        from transformer.my_transformer import MyTransformer
    print(f"Using config: {args.config}")

    # load custom vocab or GloVe
    if Config.USE_GLOVE:
        import torchtext
        glove = torchtext.vocab.GloVe(
            name='6B', dim=Config.GLOVE_EMBEDDING_SIZE,
            cache=Config.GLOVE_CACHE_DIR)
        vocab = glove
    else:
        with open(args.vocab, 'rb') as f:
            vocab = pickle.load(f)
    # load data
    df = pd.read_csv(args.csv)
    # there are some rows with label = 'label', we need to remove them
    df = df[df['label'] != 'label']
    # convert label to int
    df['label'] = df['label'].astype(np.float64)

    # split data into train, val, test (80%, 10%, 10%)
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(
        test_data, test_size=0.5, random_state=42)

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

    train_dataset = YelpReviewDataset(
        train_data, vocab, Config.MAX_SEQ_LENGTH)

    # get dataloader from dataset
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    if args.val_in_training:
        val_data = val_data.reset_index(drop=True)
        val_dataset = YelpReviewDataset(
            val_data, vocab, Config.MAX_SEQ_LENGTH)
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # define model
    if args.model_choice == 'lstm':
        model = MyLSTM(Config=Config, vocab_size=len(
            vocab), num_classes=1, device=device)
    elif args.model_choice == 'transformer':
        model = MyTransformer(Config=Config, vocab_size=len(
            vocab), output_dim=1, device=device)
    if args.load_trained:
        model_path = 'models/' + args.model_choice + '_model.pt'
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}!")
    # print num of parameters
    print(
        f'Number of trainable parameters: \
        {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.train()
    model.to(device)
    # define binary cross entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE,
                                 betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                 weight_decay=Config.WEIGHT_DECAY)

    # train model
    train(model, Config, criterion, optimizer, device, args.checkpoints, train_loader,
          val_loader if args.val_in_training else None)

    # save model
    torch.save(model.state_dict(), f'models/{args.model_choice}_model.pt')
    print(f"Model saved to models/{args.model_choice}_model.pt")
