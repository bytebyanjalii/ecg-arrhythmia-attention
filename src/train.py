# ----------------------------Step 1: Import libraries and set device---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# ----------------------------Step 2: Import your models---------------------------------
from src.models.cnn_gru import CNN_GRU
from src.models.transformer import TransformerModel

# ----------------------------Step 3: Set device---------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ----------------------------Step 4: Choose model---------------------------------
# Options: 'cnn_gru', 'transformer'
model_choice = 'transformer'  # change here to 'cnn_gru' if you want

# ----------------------------Step 5: Load CSVs and create tensors---------------------------------
train_df = pd.read_csv('data/mitdb_new/mitbih_train.csv')
test_df = pd.read_csv('data/mitdb_new/mitbih_test.csv')

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

print("DataLoaders ready. Train batches:", len(train_loader), "Test batches:", len(test_loader))

# ----------------------------Step 6: Initialize model, loss, optimizer---------------------------------
num_classes = len(set(y_train))

if model_choice == 'cnn_gru':
    model = CNN_GRU(num_classes=num_classes).to(device)
elif model_choice == 'transformer':
    model = TransformerModel(input_dim=X_train_tensor.shape[1], num_classes=num_classes).to(device)
else:
    raise ValueError("Invalid model_choice. Choose 'cnn_gru' or 'transformer'.")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Model ({model_choice}) initialized.")

# ----------------------------Step 7: Training Loop---------------------------------
num_epochs = 20
print_every = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if model_choice == 'transformer' and inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)  # transformer expects [batch, seq_len, feature_dim]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/print_every:.4f}")
            running_loss = 0.0

print("Training complete!")

# ----------------------------Step 8: Evaluation & F1 Scores---------------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if model_choice == 'transformer' and inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

from sklearn.metrics import classification_report, f1_score
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print(f"Macro F1-score: {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")
report = classification_report(y_true, y_pred)
print("\nClassification Report:\n", report)

# ----------------------------Step 9: Save Model & Results---------------------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_save_path = f"checkpoints/{model_choice}_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

results_path = f"results/{model_choice}_results.txt"
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"Macro F1-score: {f1_macro:.4f}\n")
    f.write(f"Weighted F1-score: {f1_weighted:.4f}\n\n")
    f.write(report)

print(f"Results saved to {results_path}")
