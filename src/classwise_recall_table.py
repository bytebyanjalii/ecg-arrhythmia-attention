import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.datasets import load_data
from src.models.cnn_bilstm_attn import CNN_BiLSTM_Attention

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 5
BATCH_SIZE = 512

CHECKPOINT = "checkpoints/cnn_bilstm_attn_model.pth"
SAVE_PATH = "figures/classwise_recall_table.csv"

os.makedirs("figures", exist_ok=True)

# ---------------- LOAD DATA ----------------
X_train, X_test, y_train, y_test = load_data()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = np.array(y_test)

print("Using device:", DEVICE)
print("X_test shape:", X_test_tensor.shape)

# ---------------- LOAD MODEL ----------------
model = CNN_BiLSTM_Attention(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ---------------- BATCHED PREDICTION ----------------
all_preds = []

with torch.no_grad():
    for i in range(0, X_test_tensor.shape[0], BATCH_SIZE):
        batch = X_test_tensor[i:i + BATCH_SIZE]
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())

preds = torch.cat(all_preds).numpy()

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, preds)

# ---------------- CLASS-WISE RECALL ----------------
rows = []

for i in range(NUM_CLASSES):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    recall = TP / (TP + FN + 1e-8)

    rows.append({
        "Class": f"Class {i}",
        "Recall": round(recall, 4)
    })

df = pd.DataFrame(rows)

# ---------------- SAVE ----------------
df.to_csv(SAVE_PATH, index=False)

print("\n📊 CLASS-WISE RECALL TABLE\n")
print(df)
print(f"\n✅ Saved at: {SAVE_PATH}")
