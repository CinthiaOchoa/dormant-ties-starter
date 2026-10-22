"""ReconnectScoreNet and the procedures it is compared against.

Author: Cinthia Ochoa Torre
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn


class ReconnectScoreNet(nn.Module):
    """Fully connected 4-16-8-1 network that returns one logit per pair."""

    def __init__(self, n_inputs=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1))
        # Inputs are non-negative (signals / 100); a small positive hidden bias keeps
        # ReLU units alive at the start. With default init some seeds collapsed to a
        # constant output.
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and layer.out_features > 1:
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def train_network(x_train, y_train, x_val, y_val, seed=0, lr=1e-3, batch_size=128,
                  max_epochs=500, patience=25):
    """Adam + BCE with logits; early stopping restores the epoch with the lowest validation loss.

    Inputs are expected already divided by 100.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ReconnectScoreNet(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    xv, yv = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(xt), yt).item()
            val_loss = loss_fn(model(xv), yv).item()
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'best_epoch': best_epoch, 'best_val_loss': best_loss, 'history': history}


def predict_network(model, x):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()


def fit_logistic(x_train, y_train, x_val, y_val, grid=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """L2 logistic regression; the inverse penalty C is chosen by validation average precision."""
    best = None
    for c in grid:
        model = LogisticRegression(C=c, max_iter=1000).fit(x_train, y_train)
        ap = average_precision_score(y_val, model.predict_proba(x_val)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, c, model)
    return best[2], best[1]


def synthetic_debug_matrix(n=4000, seed=0):
    """Noisy four-signal rule used only to debug the training loop, never for reported numbers."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, size=(n, 4))
    logit = 0.035 * x[:, 0] + 0.025 * x[:, 1] + 0.02 * x[:, 2] - 0.01 * x[:, 3] - 3.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(logit + rng.normal(0, 0.5, n))))).astype(float)
    return x / 100.0, y
