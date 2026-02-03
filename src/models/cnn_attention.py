import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        weights = torch.softmax(self.attn(x), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(weights * x, dim=1)       # [batch, channels]
        return context


class CNN_Attention(nn.Module):
    def __init__(self, input_dim=187, num_classes=5):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)

        # Attention
        self.attention = Attention(channels=64)

        # Classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, 187]
        x = x.unsqueeze(1)           # [batch, 1, 187]
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 93]

        x = x.permute(0, 2, 1)       # [batch, 93, 64]
        x = self.attention(x)        # [batch, 64]

        out = self.fc(x)
        return out
