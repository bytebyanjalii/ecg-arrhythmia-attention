import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """
    Load ECG data from CSV files.
    
    Returns:
        X_train: Training features (87553, 187)
        X_test: Test features (21891, 187)
        y_train: Training labels
        y_test: Test labels
    """
    data_dir = Path(__file__).parent.parent / "data" / "mitdb_new"
    
    # Load CSV files
    train_df = pd.read_csv(data_dir / "mitbih_train.csv", header=None)
    test_df = pd.read_csv(data_dir / "mitbih_test.csv", header=None)
    
    # Separate features and labels (last column is the label)
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    
    return X_train, X_test, y_train, y_test
