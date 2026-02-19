import torch
from torch.utils.data import DataLoader
from src.datasets import ECGDataset


def get_test_loader(batch_size=64):
    """
    Returns DataLoader for test dataset
    """

    test_dataset = ECGDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return test_loader

