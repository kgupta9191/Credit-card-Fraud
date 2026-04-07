# Load Modules
!module load cuda/12.6
!module load cudnn/8.9.7.29-12-cuda12.6
import torch
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import numpy as np

# Load data
data = pd.read_csv("creditcard.csv")

# Before creating tensor dataset, data cleaning is done which is shown in the report and not included in the code.
# It requires multiple plotting and generating data-charts to have a sense of outliers and data balancing.
# This required manual input. Once done this code can be easily implemented.

conv_data = torch.tensor(
    data.values,
    dtype=torch.float32
)

# Split data
X = conv_data[:, :-1]
#y = data[:, -3:].unsqueeze(1)
y = conv_data[:, -1]
dataset = TensorDataset(X, y)
N = len(dataset)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val
train_ds, val_ds, test_ds = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)  # reproducible
)

# Neural Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)




