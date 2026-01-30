import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CNN_GRU(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_GRU, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # GRU expects (batch, seq_len, features)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, 187)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)            # (batch, 64, 93)

        x = x.permute(0, 2, 1)      # (batch, 93, 64)

        _, h_n = self.gru(x)        # h_n: (1, batch, 64)
        x = h_n.squeeze(0)          # (batch, 64)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
