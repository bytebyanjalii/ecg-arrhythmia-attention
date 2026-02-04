import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader

from src.datasets import load_data
from src.models.cnn import CNN1D
from src.models.cnn_gru import CNN_GRU
from src.models.transformer import ECGTransformer
from src.models.cnn_attention import CNN_Attention
from src.models.cnn_bilstm_attn import CNN_BiLSTM_Attention

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")   # keep cpu to avoid crashes
BATCH_SIZE = 256

CKPT_DIR = "checkpoints"
FIG_DIR = "figures/confusion_matrices"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "cnn1d": (CNN1D(), "cnn1d_model.pth"),
    "cnn_gru": (CNN_GRU(), "cnn_gru_model.pth"),
    "transformer": (ECGTransformer(input_dim=187, num_classes=5), "transformer_model.pth"),
    "cnn_attention": (CNN_Attention(), "cnn_attention_model.pth"),
    "cnn_bilstm_attn": (CNN_BiLSTM_Attention(), "cnn_bilstm_attn_model.pth"),
}
# ----------------------------------------


def main():
    print(f"Using device: {DEVICE}")

    # Load test data
    _, X_test, _, y_test = load_data()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for name, (model, ckpt_name) in MODELS.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found for {name}, skipping")
            continue

        print(f"🔹 Processing {name}")

        try:
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
        except Exception as e:
            print(f"❌ Failed loading {name}: {e}")
            continue

        model.to(DEVICE)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                # shape handling
                if name in ["cnn_gru"]:
                    xb = xb.unsqueeze(1)  # (B, 1, 187)

                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[0, 1, 2, 3, 4]
        )

        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name.upper()} Confusion Matrix")
        plt.tight_layout()

        save_path = f"{FIG_DIR}/{name}_confusion.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ Saved: {save_path}")

    print("🎉 All confusion matrices generated successfully!")


if __name__ == "__main__":
    main()

