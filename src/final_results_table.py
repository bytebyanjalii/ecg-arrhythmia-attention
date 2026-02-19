import os
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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

MODELS = {
    "CNN1D": (CNN1D, "cnn1d_model.pth"),
    "CNN-GRU": (CNN_GRU, "cnn_gru_model.pth"),
    "Transformer": (ECGTransformer, "transformer_model.pth"),
    "CNN-Attention": (CNN_Attention, "cnn_attention_model.pth"),
    "CNN-BiLSTM-Attn": (CNN_BiLSTM_Attention, "cnn_bilstm_attn_model.pth"),
}

# ---------------- LOAD DATA ----------------
_, X_test, _, y_test = load_data()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_np = np.array(y_test)

# one-hot labels for AUC
y_test_bin = label_binarize(y_test_np, classes=list(range(NUM_CLASSES)))

results = []

# ---------------- LOOP OVER MODELS ----------------
for model_name, (ModelClass, ckpt_name) in MODELS.items():
    print(f"🔹 Evaluating {model_name}")

    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Missing checkpoint: {ckpt_name}")
        continue

    # Init model
    if model_name == "Transformer":
        model = ModelClass(input_dim=187, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Input shape handling
    if model_name == "CNN-GRU":
        inputs = X_test_tensor.unsqueeze(1)   # (B,1,187)
    else:
        inputs = X_test_tensor                # (B,187)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, inputs.shape[0], BATCH_SIZE):
            batch = inputs[i:i + BATCH_SIZE]
            outputs = model(batch)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()

    # Metrics
    acc = accuracy_score(y_test_np, preds)
    f1 = f1_score(y_test_np, preds, average="weighted")
    auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")

    results.append({
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro AUC": round(auc, 4),
        "Weighted F1": round(f1, 4)
    })

# ---------------- FINAL TABLE ----------------
df = pd.DataFrame(results)
print("\n📊 FINAL RESULTS TABLE\n")
print(df.to_string(index=False))

# Optional: save to CSV
df.to_csv("final_results_table.csv", index=False)
print("\n✅ Saved as final_results_table.csv")
