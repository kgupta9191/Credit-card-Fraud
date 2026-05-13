import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset_from_dataframe(dataframe, target_col=-1):
    """Convert a dataframe into a TensorDataset and return labels separately.

    target_col identifies the label column (default -1 means last column).
    Returns (dataset, labels_tensor).
    """
    conv_data = torch.tensor(dataframe.values, dtype=torch.float32)
    num_cols = conv_data.shape[1]
    target_idx = target_col if target_col >= 0 else num_cols + target_col
    if target_idx < 0 or target_idx >= num_cols:
        raise IndexError(f"target_col {target_col} is out of range for {num_cols} columns")
    y = conv_data[:, target_idx]
    X = torch.cat((conv_data[:, :target_idx], conv_data[:, target_idx + 1 :]), dim=1)
    return TensorDataset(X, y), y


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def get_pos_weight(labels):
    labels = labels.float()
    positives = labels.sum().item()
    negatives = len(labels) - positives
    if positives <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def train_step(model, batch, criterion, optimizer, device):
    xb, yb = batch
    xb = xb.to(device)
    yb = yb.to(device).float().unsqueeze(1)
    optimizer.zero_grad()
    pred = model(xb)
    loss = criterion(pred, yb)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)
            total_loss += criterion(model(xb), yb).item()
            num_batches += 1
    return total_loss / max(1, num_batches)


def main(
    data_path="creditcard.csv",
    epochs=500,
    eval_every=100,
    patience=500,
    batch_size=1024,
    num_workers=4,
    input_dim=28,
    hidden_dim=128,
    enable_compile=True,
    compile_backend="inductor",
):
    data = pd.read_csv(data_path)
    dataset, all_labels = build_dataset_from_dataframe(data)
    train_ds, val_ds, test_ds = split_dataset(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    if enable_compile and hasattr(torch, "compile"):
        model = torch.compile(model, backend=compile_backend)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if hasattr(train_ds, "indices"):
        train_indices = torch.tensor(train_ds.indices, dtype=torch.long)
        train_labels = all_labels[train_indices]
    else:
        train_labels = torch.tensor([label.item() for _, label in train_ds], dtype=torch.float32)
    pos_weight = get_pos_weight(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val = float("inf")
    wait = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            train_loss += train_step(model, batch, criterion, optimizer, device)
            batches += 1
        avg_train_loss = train_loss / max(1, batches)

        if epoch % eval_every == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4e} Val Loss = {val_loss:.4e}")

            if val_loss < best_val:
                best_val = val_loss
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), "best_model.pth")
                wait = 0
            else:
                wait += eval_every

            if wait >= patience:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss = {test_loss:.4e}")

    return model


if __name__ == "__main__":
    main()
