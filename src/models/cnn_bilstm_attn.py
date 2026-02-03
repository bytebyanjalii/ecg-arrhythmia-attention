import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        weights = torch.softmax(self.attn(x), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(weights * x, dim=1)       # [batch, hidden_dim]
        return context


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim=187, num_classes=5):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = Attention(hidden_dim=128)

        # Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, 187]
        x = x.unsqueeze(1)               # [batch, 1, 187]
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 93]

        x = x.permute(0, 2, 1)           # [batch, 93, 64]
        lstm_out, _ = self.lstm(x)       # [batch, 93, 128]

        context = self.attention(lstm_out)
        out = self.fc(context)
        return out
