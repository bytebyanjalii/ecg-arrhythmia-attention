import pandas as pd
import torch
from src.models.cnn import CNN1D  # Correct class name in cnn.py

# -------------------------------
# 1️⃣ Paths to dataset (from project root)
# -------------------------------
train_csv = 'data/mitdb_new/mitbih_train.csv'
test_csv  = 'data/mitdb_new/mitbih_test.csv'

# -------------------------------
# 2️⃣ Load datasets
# -------------------------------
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# -------------------------------
# 3️⃣ Split features and labels
# Assuming last column is label
# -------------------------------
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# -------------------------------
# 4️⃣ Convert to PyTorch tensors
# -------------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# -------------------------------
# 5️⃣ Initialize CNN model
# -------------------------------
model = CNN1D(input_dim=X_train_tensor.shape[1], num_classes=len(set(y_train)))
print(model)

# -------------------------------
# ✅ Ready to train!
# -------------------------------

