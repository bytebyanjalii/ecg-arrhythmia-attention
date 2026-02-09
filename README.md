ECG Arrhythmia Detection Using Attention-Based Deep Learning

1. Introduction
Cardiovascular diseases are among the leading causes of mortality worldwide. Electrocardiograms (ECGs) are widely used to diagnose heart-related disorders, especially arrhythmias. Manual ECG analysis is time-consuming, requires expert knowledge, and may vary between clinicians.  
This project focuses on building an automated deep learning–based system to classify ECG heartbeats into different arrhythmia classes with high accuracy and reliability.

2. Objective
The main goals of this project are:
- Automated detection and classification of ECG arrhythmias
- Comparison of multiple deep learning architectures
- Performance evaluation using standard medical ML metrics
- Use of attention mechanisms to improve interpretability

3. Dataset Description
- Dataset: MIT-BIH Arrhythmia Dataset
- Signal type: 1D ECG signals
- Each heartbeat length: 187 samples
- Number of classes: 5 arrhythmia categories
- Task type: Multi-class classification

Each ECG signal is preprocessed, normalized, segmented into fixed-length heartbeats, and then split into training and testing sets.

4. Overall Workflow
ECG Signal Input  
→ Preprocessing & Normalization  
→ Heartbeat Segmentation (187 samples)  
→ Train–Test Split  
→ Deep Learning Model Training  
→ Prediction  
→ Evaluation & Visualization  

5. Models Implemented

5.1 CNN1D
- Uses 1D convolution layers
- Learns local ECG patterns like QRS complexes
- Serves as a baseline deep learning model

5.2 CNN-GRU
- CNN extracts spatial ECG features
- GRU captures temporal dependencies
- Improves sequence understanding over CNN alone

5.3 Transformer
- Uses self-attention instead of recurrence
- Captures long-range dependencies in ECG signals
- Effective for global feature learning

5.4 CNN-Attention
- CNN extracts features
- Attention layer highlights important time steps
- Improves model interpretability

5.5 CNN-BiLSTM-Attention
- CNN for feature extraction
- BiLSTM captures forward and backward temporal context
- Attention focuses on clinically important ECG regions
- Best-performing model in this project

6. Project Structure



7. Evaluation Metrics
The models are evaluated using:
- Accuracy
- Macro ROC-AUC
- Weighted F1-score
- Class-wise Recall
- Confusion Matrix
- ROC Curves for each model

8. Final Results Summary

Model               Accuracy   Macro AUC   Weighted F1  
CNN1D               0.9816     0.9871      0.9810  
CNN-GRU             0.9784     0.9855      0.9775  
Transformer         0.9762     0.9874      0.9755  
CNN-Attention       0.9542     0.9569      0.9492  
CNN-BiLSTM-Attn     0.9820     0.9917      0.9813  

9. Attention Weight Visualization
Attention-based models generate attention weight plots that show which parts of the ECG signal contributed most to the prediction. This:
- Improves explainability (XAI)
- Helps understand model decisions
- Acts as a key differentiator from standard CNN models

10. How to Run the Project

Install dependencies:
pip install -r requirements.txt

Train models:
python -m src.train

Generate evaluation outputs:
python -m src.plot_confusion_matrices
python -m src.plot_roc_curves
python -m src.classwise_recall_table
python -m src.final_results_table
python -m src.plot_attention_weights

11. Conclusion
This project demonstrates that deep learning models, especially attention-based architectures, can achieve high accuracy in ECG arrhythmia classification. The CNN-BiLSTM-Attention model performed best, offering both strong predictive performance and interpretability, making it suitable for real-world clinical decision support systems.
