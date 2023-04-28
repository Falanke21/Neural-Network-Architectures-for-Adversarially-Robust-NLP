import torch


class MyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, device):
        super(MyLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        # x shape (batch_size, seq_length, vocab_size)
        x = x.long()
        x = self.embedding(x)  # (batch_size, seq_length, embedding_size)

        # h0, c0 shape (num_layers*2, batch_size, hidden_size)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out shape (batch_size, hidden_size*2)
        out = self.fc(out)  # (batch_size, num_classes)
        return out

# TODO LSTM with attention?
