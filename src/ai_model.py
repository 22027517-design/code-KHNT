import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class RiskPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_risk_model(X, y, in_dim, epochs=50):
    model = RiskPredictor(in_dim)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def infer_risk(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).cpu().numpy().reshape(-1)