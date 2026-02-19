import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import load_data
from src.models.cnn_attention import CNN_Attention

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 5
CKPT_PATH = "checkpoints/cnn_attention_model.pth"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
_, X_test, _, y_test = load_data()

# take ONE ECG sample
sample_idx = 0
ecg_signal = X_test[sample_idx]          # shape (187,)

# ✅ CORRECT SHAPE: (1, 1, 187)
X_tensor = torch.tensor(ecg_signal, dtype=torch.float32)\
                .unsqueeze(0)\
                .unsqueeze(1)\
                .to(DEVICE)

print("Input shape:", X_tensor.shape)

# ---------------- LOAD MODEL ----------------
model = CNN_Attention(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ---------------- FORWARD ----------------
with torch.no_grad():
    logits, attn_weights = model(X_tensor)

# attn_weights: (1, 187)
attn_weights = attn_weights.squeeze().cpu().numpy()

# ---------------- PLOT ----------------
plt.figure(figsize=(12, 4))

plt.plot(ecg_signal, label="ECG Signal", color="black")

plt.fill_between(
    range(len(attn_weights)),
    np.min(ecg_signal),
    np.max(ecg_signal),
    where=attn_weights > np.mean(attn_weights),
    alpha=0.35,
    label="High Attention"
)

plt.title("Attention Weight Visualization on ECG Signal")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()

save_path = os.path.join(SAVE_DIR, "attention_visualization.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Saved attention visualization at: {save_path}")
