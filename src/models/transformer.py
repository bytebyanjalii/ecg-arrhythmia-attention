import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Input linear layer to map input_dim to d_model
        self.input_fc = nn.Linear(1, d_model)  # because feature_dim = 1

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # important for [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len] --> expand to [batch, seq_len, 1] if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        x = self.input_fc(x)          # [batch, seq_len, d_model]
        x = self.pos_encoder(x)       # add positional encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Take mean over sequence dimension
        x = x.mean(dim=1)             # [batch, d_model]

        out = self.fc_out(x)          # [batch, num_classes]
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Alias for compatibility
ECGTransformer = TransformerModel