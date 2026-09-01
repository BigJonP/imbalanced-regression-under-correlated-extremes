"""One shared model for every method: MLP, Adam, early stopping.

Methods differ only in sample weights, resampled data, or the loss options
below; architecture, optimizer, and budget stay identical. Targets are
standardized internally for optimization and inverted at predict time.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dire.data.panel import TARGET_LOG, TARGET_RAW, feature_columns
from dire.eval.fold_stats import N_BINS, market_scaler
from dire.seeding import set_all_seeds


class _Net(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def _st_rank(m):
    # straight-through ranks in [0, 1]: rank values forward, identity gradient back
    ranks = m.argsort(dim=-1).argsort(dim=-1).float() / max(m.shape[-1] - 1, 1)
    return m + (ranks - m).detach()


def _ranksim_loss(features, y):
    """Gong et al. (2022): feature-space similarity rankings should match
    label-space similarity rankings, per batch row."""
    sf = -torch.cdist(features, features)
    sy = -torch.cdist(y.unsqueeze(1), y.unsqueeze(1))
    return F.mse_loss(_st_rank(sf), _st_rank(sy))


class _FDS:
    """Yang et al. (2021) feature distribution smoothing, diagonal-covariance
    variant: per-target-bin running mean/var of penultimate features, kernel
    smoothed across bins; features are whiten-recolored during training from
    the second epoch on."""

    def __init__(self, n_feat, edges, device, kernel_size=5, sigma=2.0, momentum=0.9):
        self.edges = edges
        self.momentum = momentum
        n_bins = len(edges) - 1
        self.mu = torch.zeros(n_bins, n_feat, device=device)
        self.var = torch.ones(n_bins, n_feat, device=device)
        self.smu = self.mu.clone()
        self.svar = self.var.clone()
        x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        k = torch.exp(-(x**2) / (2.0 * sigma**2))
        self._kernel = (k / k.sum()).view(1, 1, -1)
        self.started = False

    def bin_of(self, y):
        return torch.clamp(torch.bucketize(y, self.edges) - 1, 0, len(self.edges) - 2)

    def _smooth(self, m):
        pad = self._kernel.shape[-1] // 2
        num = F.conv1d(m.T.unsqueeze(1), self._kernel, padding=pad)
        den = F.conv1d(torch.ones_like(m.T.unsqueeze(1)), self._kernel, padding=pad)
        return (num / den).squeeze(1).T

    @torch.no_grad()
    def update(self, feats, bins):
        for b in bins.unique():
            rows = feats[bins == b]
            m, v = rows.mean(0), rows.var(0, unbiased=False)
            if self.started:
                self.mu[b] = self.momentum * self.mu[b] + (1 - self.momentum) * m
                self.var[b] = self.momentum * self.var[b] + (1 - self.momentum) * v
            else:
                self.mu[b], self.var[b] = m, v
        self.smu, self.svar = self._smooth(self.mu), self._smooth(self.var)
        self.started = True

    def calibrate(self, feats, bins):
        if not self.started:
            return feats
        mu, var = self.mu[bins], self.var[bins]
        smu, svar = self.smu[bins], self.svar[bins]
        return (feats - mu) / torch.sqrt(var + 1e-6) * torch.sqrt(svar + 1e-6) + smu


class MLPRegressor:
    def __init__(self, seed=0, target="raw", hidden=(64, 64), lr=1e-3, batch_size=256,
                 max_epochs=100, patience=10, loss="mse", ranksim_lambda=0.0, fds=False,
                 device=None):
        if target not in ("raw", "log") or loss not in ("mse", "bmc"):
            raise ValueError("target must be raw|log, loss must be mse|bmc")
        self.seed, self.target, self.hidden, self.lr = seed, target, tuple(hidden), lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.loss, self.ranksim_lambda, self.fds = loss, ranksim_lambda, fds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _design(self, df):
        X = df[self._features].to_numpy(dtype=np.float32)
        return (X - self._f_mean) / self._f_sd

    def _targets(self, df):
        col = TARGET_LOG if self.target == "log" else TARGET_RAW
        return df[col].to_numpy(dtype=np.float32)

    def fit(self, train_df, val_df=None, sample_weight=None):
        set_all_seeds(self.seed)
        dev = self.device
        self._features = feature_columns(train_df)
        scaler = market_scaler(train_df).astype(np.float32)
        self._f_mean, self._f_sd = scaler[0], np.where(scaler[1] == 0, 1.0, scaler[1])

        t = self._targets(train_df)
        self._t_mean, self._t_sd = float(t.mean()), float(t.std()) or 1.0
        X = torch.tensor(self._design(train_df), device=dev)
        y = torch.tensor((t - self._t_mean) / self._t_sd, device=dev)
        if sample_weight is None:
            w = torch.ones(len(X), device=dev)
        else:
            sw = np.asarray(sample_weight, dtype=np.float32)
            w = torch.tensor(sw / sw.mean(), device=dev)

        net = _Net(len(self._features), self.hidden).to(dev)
        params = list(net.parameters())
        if self.loss == "bmc":
            self._log_noise = torch.zeros(1, device=dev, requires_grad=True)
            params.append(self._log_noise)
        opt = torch.optim.Adam(params, lr=self.lr)

        fds = None
        if self.fds:
            edges = torch.linspace(float(y.min()), float(y.max()) + 1e-6, N_BINS + 1, device=dev)
            fds = _FDS(self.hidden[-1], edges, dev)
            fds_bins = fds.bin_of(y)

        if val_df is not None:
            Xv = torch.tensor(self._design(val_df), device=dev)
            yv = torch.tensor((self._targets(val_df) - self._t_mean) / self._t_sd, device=dev)
        gen = torch.Generator().manual_seed(self.seed)
        best_loss, best_state, bad = float("inf"), None, 0

        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(X), generator=gen).to(dev)
            epoch_feats, epoch_bins = [], []
            for start in range(0, len(X), self.batch_size):
                b = perm[start : start + self.batch_size]
                z = net.body(X[b])
                z_cal = fds.calibrate(z, fds_bins[b]) if fds is not None else z
                pred = net.head(z_cal).squeeze(-1)
                if self.loss == "mse":
                    batch_loss = (w[b] * (pred - y[b]) ** 2).mean()
                else:
                    noise_var = torch.exp(self._log_noise)
                    logits = -(pred.unsqueeze(1) - y[b].unsqueeze(0)) ** 2 / (2 * noise_var)
                    batch_loss = F.cross_entropy(
                        logits, torch.arange(len(b), device=dev)
                    ) * (2 * noise_var).detach()
                if self.ranksim_lambda:
                    batch_loss = batch_loss + self.ranksim_lambda * _ranksim_loss(z_cal, y[b])
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                if fds is not None:
                    epoch_feats.append(z.detach())
                    epoch_bins.append(fds_bins[b])
            if fds is not None:
                fds.update(torch.cat(epoch_feats), torch.cat(epoch_bins))
            self.n_epochs_ = epoch + 1
            if val_df is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(F.mse_loss(net(Xv), yv))
                if val_loss < best_loss - 1e-7:
                    best_loss, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net.eval()
        return self

    def predict(self, df):
        X = torch.tensor(self._design(df), device=self.device)
        with torch.no_grad():
            p = self._net(X).cpu().numpy()
        p = p * self._t_sd + self._t_mean
        return np.exp(p) if self.target == "log" else p
