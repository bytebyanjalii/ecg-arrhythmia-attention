import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import os

# =======================
# IMPORT MODELS
# =======================
from src.models.cnn import CNN1D
from src.models.cnn_gru import CNN_GRU
from src.models.transformer import TransformerModel
from src.models.cnn_bilstm_attn import CNN_BiLSTM_Attention
from src.models.cnn_attention import CNN_Attention

# =======================
# CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

epochs = 20
batch_size = 64
learning_rate = 1e-3

# 👉 CHANGE MODEL HERE
model_choice = "cnn_attention"
# options:
# cnn1d | cnn_gru | transformer | cnn_bilstm_attn | cnn_attention

# =======================
# LOAD DATA
# =======================
train_df = pd.read_csv("data/mitdb_new/mitbih_train.csv")
test_df = pd.read_csv("data/mitdb_new/mitbih_test.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(np.unique(y_train))
print("DataLoaders ready")

# =======================
# MODEL SELECTION
# =======================
if model_choice == "cnn1d":
    model = CNN1D(X_train.shape[1], num_classes)

elif model_choice == "cnn_gru":
    model = CNN_GRU(num_classes=num_classes)

elif model_choice == "transformer":
    model = TransformerModel(X_train.shape[1], num_classes)

elif model_choice == "cnn_bilstm_attn":
    model = CNN_BiLSTM_Attention(X_train.shape[1], num_classes)

elif model_choice == "cnn_attention":
    model = CNN_Attention(X_train.shape[1], num_classes)

else:
    raise ValueError("Invalid model choice")

model.to(device)
print(f"Model initialized: {model_choice}")

# =======================
# TRAINING SETUP
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =======================
# TRAIN LOOP
# =======================
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(train_loader):.4f}")

print("Training complete!")

# =======================
# EVALUATION
# =======================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print("\nTest Accuracy:", acc)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =======================
# SAVE MODEL & RESULTS
# =======================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), f"checkpoints/{model_choice}_model.pth")

with open(f"results/{model_choice}_results.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n\n")
    f.write(classification_report(y_true, y_pred))

print(f"\n✅ Model & results saved for: {model_choice}")
