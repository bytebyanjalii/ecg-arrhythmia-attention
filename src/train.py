import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.datasets import load_data
from src.models.cnn1d import CNN1D
from src.models.cnn_gru import CNN_GRU
from src.models.transformer import ECGTransformer
from src.models.cnn_bilstm_attn import CNN_BiLSTM_Attention


# =======================
# CONFIG
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
learning_rate = 1e-3
batch_size = 64

model_choice = "cnn_bilstm_attn"
# "cnn1d", "cnn_gru", "transformer", "cnn_bilstm_attn"


# =======================
# LOAD DATA
# =======================
X_train, X_test, y_train, y_test = load_data()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(y_train))


# =======================
# MODEL SELECTION
# =======================
if model_choice == "cnn1d":
    model = CNN1D(input_dim=X_train.shape[1], num_classes=num_classes)

elif model_choice == "cnn_gru":
    model = CNN_GRU(input_dim=X_train.shape[1], num_classes=num_classes)

elif model_choice == "transformer":
    model = ECGTransformer(input_dim=X_train.shape[1], num_classes=num_classes)

elif model_choice == "cnn_bilstm_attn":
    model = CNN_BiLSTM_Attention(num_classes=num_classes)

else:
    raise ValueError("Invalid model choice")

model.to(device)


# =======================
# TRAINING
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")


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
# SAVE
# =======================
torch.save(model.state_dict(), f"checkpoints/{model_choice}_model.pth")

with open(f"results/{model_choice}_results.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n\n")
    f.write(classification_report(y_true, y_pred))

print(f"\n✅ Saved model & results for {model_choice}")

