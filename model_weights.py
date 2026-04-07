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

# Model, Loss function and Optmizer
model = MLP()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
pos_weight = torch.tensor([284315 / 492]).to(device)  # weight = N_neg / N_pos
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.compile(model, backend="inductor")  # or backend="nvfuser" on GPU
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=1024)

# Deploying Model
epochs = 500
patience = 500
best_val = float('inf')
wait = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yb = yb.float().unsqueeze(1)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                yb = yb.float().unsqueeze(1)
                val_loss = criterion(model(xb), yb).item()

        print(f"Epoch {epoch}: Train Loss = {loss:.4e} Val Loss = {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 100

        if wait >= patience:
            print("Early stopping triggered")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)



