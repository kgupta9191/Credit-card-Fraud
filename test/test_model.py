import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLP, evaluate, get_pos_weight, split_dataset, train_step

DEFAULT_POS_WEIGHT = torch.tensor([1.0])


def test_mlp_forward_shape():
    model = MLP(input_dim=28, hidden_dim=32)
    batch = torch.randn(16, 28)
    output = model(batch)
    assert output.shape == (16, 1)


def test_split_dataset_preserves_all_records():
    x = torch.randn(100, 28)
    y = torch.randint(0, 2, (100,), dtype=torch.float32)
    dataset = TensorDataset(x, y)
    train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1)
    assert len(train_ds) == 80
    assert len(val_ds) == 10
    assert len(test_ds) == 10
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(dataset)


def test_train_step_updates_model_parameters():
    torch.manual_seed(1)
    model = MLP(input_dim=28, hidden_dim=16)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=DEFAULT_POS_WEIGHT)
    xb = torch.randn(32, 28)
    yb = torch.randint(0, 2, (32,), dtype=torch.float32)
    before = [param.detach().clone() for param in model.parameters()]
    loss = train_step(model, (xb, yb), criterion, optimizer, device)
    after = [param.detach() for param in model.parameters()]
    assert loss >= 0
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_returns_valid_loss():
    model = MLP(input_dim=28, hidden_dim=16)
    criterion = nn.BCEWithLogitsLoss(pos_weight=DEFAULT_POS_WEIGHT)
    x = torch.randn(64, 28)
    y = torch.randint(0, 2, (64,), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    loss = evaluate(model, loader, criterion, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0


def test_pos_weight_handles_no_positive_labels():
    labels = torch.zeros(10)
    pos_weight = get_pos_weight(labels)
    assert torch.allclose(pos_weight, torch.tensor([1.0]))
