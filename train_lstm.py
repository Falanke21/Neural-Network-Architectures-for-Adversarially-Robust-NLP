import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from lstm.my_lstm import MyLSTM
from utils.tokenizer import tokenize


USE_GPU = True
SEQ_LENGTH = 200
BATCH_SIZE = 32
LSTM_HIDDEN_SIZE = 128
LSTM_EMBEDDING_SIZE = 128
LSTM_NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

device = torch.device(
    'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device index: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')


class YelpReviewDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']  # text is a string
        token_list = tokenize(text)
        indices = torch.zeros(SEQ_LENGTH, dtype=torch.long)  # initialize as 0s
        for i, token in enumerate(token_list):
            if i >= SEQ_LENGTH:
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
    # load vocab
    with open('data/vocab10k.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # load data
    df = pd.read_csv('data/data10k.csv')
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    # Reset dataframe index so that we can use df.loc[idx, 'text']
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_dataset = YelpReviewDataset(train_data, vocab)
    test_dataset = YelpReviewDataset(test_data, vocab)

    # get dataloader from dataset
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # define lstm model
    model = MyLSTM(vocab_size=len(vocab), embedding_size=LSTM_EMBEDDING_SIZE,
                   hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, 
                   num_classes=1, device=device)
    # print num of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    model.train()
    model.to(device)
    # define binary cross entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # training loop
    print("Training...")
    for epoch in range(NUM_EPOCHS):
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
            if (i+1) % 20 == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, \
                    Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {total_loss / len(train_loader):.4f}")
        # save checkpoint
        torch.save(model.state_dict(), f'models/checkpoints/lstm_model_epoch{epoch+1}.pt')
                
    # save model
    torch.save(model.state_dict(), 'models/lstm_model.pt')

    # load model
    model.load_state_dict(torch.load('models/lstm_model.pt'))

    # test on test set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        print("Testing...")
        for data, labels in tqdm(test_loader):
            data = data.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {correct / total:.4f}")
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
