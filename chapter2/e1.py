import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

digitsx = load_digits(v