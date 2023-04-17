import torch

USE_GPU = False
device = torch.device(
    'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch_size, seq_length, input_size)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
