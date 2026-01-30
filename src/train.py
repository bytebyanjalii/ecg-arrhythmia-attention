# ----------------------------Step 1: Import libraries and set device---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from src.models.cnn import CNN1D  # import our CNN model



# ----------------------------------------Step 2: Set device--------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# --------------------------------------Step 3: Load CSVs and create tensors---------------------------------
# Load CSV files
train_df = pd.read_csv('data/mitdb_new/mitbih_train.csv')
test_df = pd.read_csv('data/mitdb_new/mitbih_test.csv')

# Split features and labels
X_train = train_df.iloc[:, :-1].values  # all columns except last
y_train = train_df.iloc[:, -1].values   # last column as label

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # long for classification

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoaders ready. Train batches:", len(train_loader), "Test batches:", len(test_loader))




# -----------------------------------------------Step 4: Model, Loss, Optimizer----------------------------------
# Import your model at the top if not already
# from models.cnn import CNN  
# # or whichever model you want to train

# Initialize the model
model = CNN1D(input_dim=X_train_tensor.shape[1], num_classes=len(set(y_train)))   # adjust num_classes
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()  # suitable for classification

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model, loss function, and optimizer initialized.")




# -------------------------------------------------Step 5: Training Loop----------------------
# Training parameters
num_epochs = 20  # you can adjust
print_every = 100  # how often to print loss

# Training loop
for epoch in range(num_epochs):
    model.train()  # set model to training mode
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print loss periodically
        if (i + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/print_every:.4f}")
            running_loss = 0.0

print("Training complete!")




# ---------------------------------------------------Step 6: Validation / Testing & Save Model-------------------------
# Set model to evaluation mode
model.eval()

# Disable gradient computation for validation/testing
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model
model_save_path = "checkpoints/final_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")



from sklearn.metrics import classification_report, f1_score
import os

print(f"Test Accuracy: {accuracy:.2f}%")
# ===== F1 SCORE CALCULATION =====
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# F1 Scores
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Macro F1-score: {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

# Full classification report
report = classification_report(y_true, y_pred)
print("\nClassification Report:\n", report)

# ===== SAVE RESULTS =====
os.makedirs("results", exist_ok=True)

with open("results/cnn1d_results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"Macro F1-score: {f1_macro:.4f}\n")
    f.write(f"Weighted F1-score: {f1_weighted:.4f}\n\n")
    f.write(report)

