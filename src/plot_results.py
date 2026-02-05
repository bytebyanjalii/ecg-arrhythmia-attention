import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
FIG_DIR = "figures/model_comparison"

os.makedirs(FIG_DIR, exist_ok=True)

models = [
    "cnn1d",
    "cnn_gru",
    "transformer",
    "cnn_attention",
    "cnn_bilstm_attn"
]

accuracies = []
macro_f1 = []
weighted_f1 = []
used_models = []

def extract_metric(text, key, percent=False):
    for line in text.splitlines():
        if key in line:
            value = line.split(":")[1].strip().replace("%", "")
            value = float(value)
            return value / 100 if percent else value
    return None

for model in models:
    path = os.path.join(RESULTS_DIR, f"{model}_results.txt")

    if not os.path.exists(path):
        print(f"Skipping {model} (file missing)")
        continue

    with open(path, "r") as f:
        text = f.read()

    acc = extract_metric(text, "Test Accuracy", percent=True)
    mf1 = extract_metric(text, "Macro F1-score")
    wf1 = extract_metric(text, "Weighted F1-score")

    if None in (acc, mf1, wf1):
        print(f"⚠️ Incomplete metrics in {model}, skipping")
        continue

    used_models.append(model)
    accuracies.append(acc)
    macro_f1.append(mf1)
    weighted_f1.append(wf1)

x = np.arange(len(used_models))
w = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - w, accuracies, w, label="Accuracy")
plt.bar(x, macro_f1, w, label="Macro F1")
plt.bar(x + w, weighted_f1, w, label="Weighted F1")

plt.xticks(x, used_models, rotation=15)
plt.ylabel("Score")
plt.title("ECG Arrhythmia Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(f"{FIG_DIR}/model_comparison.png", dpi=300)
plt.close()

print("✅ Model comparison graph saved successfully")
