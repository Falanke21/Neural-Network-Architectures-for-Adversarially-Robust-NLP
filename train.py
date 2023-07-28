import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.model_factory import construct_model_from_config
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


def train(model, Config, criterion, optimizer, device, checkpoints, train_loader, val_loader):
    print("Training...")
    train_losses, val_losses, val_accuracy = [], [], []
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        model.train()
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.unsqueeze(1).float()  # (batch_size, 1)
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

            # update tqdm with loss value every a few batches
            if (i+1) % (Config.BATCH_SIZE * 3) == 0:
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
                checkpoint_path = f'{args.output_dir}/checkpoints/{os.environ["MODEL_CHOICE"]}_model_epoch{epoch+1}.pt'
                torch.save(model.state_dict(), checkpoint_path)
            except OSError as e:
                print(f"Could not save checkpoint at epoch {epoch+1}, error: {e}")

        # evaluate on validation set if necessary
        model.eval()
        with torch.no_grad():
            total_loss = total = TP = TN = 0
            print(f"Validation at epoch {epoch + 1}...")
            for data, labels in tqdm(val_loader):
                data = data.to(device)
                labels = labels.unsqueeze(1).float().to(device)
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

        # plot loss and accuracy values to file
        if args.loss_values:
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_train_losses.txt', 'a') as f:
                f.write(f'{train_losses[-1]}\n')
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_losses.txt', 'a') as f:
                f.write(f'{val_losses[-1]}\n')
            with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt', 'a') as f:
                f.write(f'{val_accuracy[-1]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--load-trained', action='store_true', default=False)
    parser.add_argument('--output-dir', type=str, default='tmp')
    parser.add_argument('--checkpoints', action='store_true', default=False)
    parser.add_argument('--loss-values', action='store_true', default=False, help='Output txt files of loss values')
    args = parser.parse_args()

    # default config file to output_dir/config.py
    config_path = f'{args.output_dir}/config.py'
   
    # create output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.checkpoints:
        if not os.path.exists(f'{args.output_dir}/checkpoints'):
            os.makedirs(f'{args.output_dir}/checkpoints')
    print(f'Will write outputs to "{args.output_dir}"')

    # Constructing model...
    model, Config, vocab, device = construct_model_from_config(config_path)

    if args.load_trained:
        model_path = f'{args.output_dir}/' + os.environ["MODEL_CHOICE"] + '_model.pt'
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}!")
    # print num of parameters
    print(
        f'Number of trainable parameters: \
        {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model.train()
    model.to(device)

    # load data
    train_data = pd.read_csv(f'{args.csv_folder}/train.csv')
    val_data = pd.read_csv(f'{args.csv_folder}/val.csv')
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
    val_data = val_data.reset_index(drop=True)
    val_dataset = YelpReviewDataset(
        val_data, vocab, Config.MAX_SEQ_LENGTH)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # define binary cross entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if hasattr(Config, 'USE_ADAMW') and Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                      betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                      weight_decay=Config.WEIGHT_DECAY)
        print("Using AdamW optimizer")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE,
                                    betas=Config.BETAS, eps=Config.ADAM_EPSILON,
                                    weight_decay=Config.WEIGHT_DECAY)

    # train model
    train(model, Config, criterion, optimizer, device, args.checkpoints, train_loader,
          val_loader)

    # save model
    torch.save(model.state_dict(), f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_model.pt')
    print(f'Model saved to {args.output_dir}/{os.environ["MODEL_CHOICE"]}_model.pt')
