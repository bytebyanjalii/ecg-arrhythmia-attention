
# src/plot_roc_curves.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from src.datasets import load_data
from src.models.cnn import CNN1D
from src.models.cnn_gru import CNN_GRU
from src.models.transformer import ECGTransformer
from src.models.cnn_attention import CNN_Attention
from src.models.cnn_bilstm_attn import CNN_BiLSTM_Attention

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 5
BATCH_SIZE = 512

CKPT_DIR = "checkpoints"
FIG_DIR = "figures/roc_curves"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "cnn1d": (CNN1D, "cnn1d_model.pth"),
    "cnn_gru": (CNN_GRU, "cnn_gru_model.pth"),
    "transformer": (ECGTransformer, "transformer_model.pth"),
    "cnn_attention": (CNN_Attention, "cnn_attention_model.pth"),
    "cnn_bilstm_attn": (CNN_BiLSTM_Attention, "cnn_bilstm_attn_model.pth"),
}

# ---------------- LOAD DATA ----------------
X_train, X_test, y_train, y_test = load_data()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_np = y_test

y_test_bin = label_binarize(y_test_np, classes=list(range(NUM_CLASSES)))

print("Using device:", DEVICE)
print("X_test_tensor shape:", X_test_tensor.shape)

# ---------------- LOOP OVER MODELS ----------------
for model_name, (ModelClass, ckpt_name) in MODELS.items():
    print(f"\n🔹 Processing {model_name}")

    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Missing checkpoint: {ckpt_name}")
        continue

    # ---- INIT MODEL ----
    if model_name == "transformer":
        model = ModelClass(input_dim=187, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    all_probs = []

    with torch.no_grad():
        for i in range(0, X_test_tensor.shape[0], BATCH_SIZE):
            batch = X_test_tensor[i:i + BATCH_SIZE]

            # 🚨 SHAPE FIX (THIS IS THE KEY)
            if model_name == "cnn_gru":
                batch = batch.unsqueeze(1)   # (B, 1, 187)

            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()

    # ---------------- ROC ----------------
    plt.figure(figsize=(6, 5))

    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")

    save_path = os.path.join(FIG_DIR, f"{model_name}_roc.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}")

print("\n🎉 ALL ROC CURVES DONE SUCCESSFULLY")
