# -*- coding: utf-8 -*-
"""LSTM et Transformer sur la serie ré-échantillonnée. Sortie = incréments de perte cumulés."""
import numpy as np

from .base import CYCLES, N_CYCLES, SEED, Candidate, cond_features, cycle_features


def _torch():
    import torch
    return torch


class _SequenceCandidate(Candidate):
    """Partie commune : grille, normalisation, entraînement, prédiction."""

    name = "sequence"
    needs = ("torch",)

    def __init__(self, n_steps=240, hidden=64, epochs=5000, lr=0.01,
                 weight_decay=1e-4, autoregressive=False):
        self.n_steps = n_steps
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.autoregressive = autoregressive

    # ------------------------------------------------------------ données
    def _grid(self):
        """Cycle au centre de chaque bloc de la grille régulière."""
        edges = np.linspace(0, N_CYCLES, self.n_steps + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def _inputs(self, T, C):
        """Séquence d'entrée (n_steps, n_features) pour une condition donnée."""
        grid = self._grid()
        cyc = cycle_features(grid)
        cond = np.repeat(cond_features([T], [C]), len(grid), axis=0)
        return np.column_stack([cyc, cond]).astype(np.float32)

    def _targets(self, curves):
        """Perte de SOH moyenne par bloc + masque des blocs observés."""
        grid_idx = np.clip(((np.arange(N_CYCLES) ) * self.n_steps) // N_CYCLES,
                           0, self.n_steps - 1)
        Y = np.full((len(curves), self.n_steps), np.nan, dtype=np.float32)
        for k, (_, _, n, y) in enumerate(curves):
            b = grid_idx[np.clip(n.astype(int), 1, N_CYCLES) - 1]
            loss = self.a_ - y
            somme = np.bincount(b, weights=loss, minlength=self.n_steps)
            compte = np.bincount(b, minlength=self.n_steps)
            vus = compte > 0
            Y[k, vus] = (somme[vus] / compte[vus]).astype(np.float32)
        return Y

    # ------------------------------------------------- entraînement commun
    def _fit(self, curves):
        torch = _torch()
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        X = np.stack([self._inputs(T, C) for T, C, _, _ in curves])       # (B, L, F)
        Y = self._targets(curves)                                          # (B, L)
        mask = np.isfinite(Y)
        self.echelle_ = float(max(np.nanmax(Y), 1.0))
        Yn = np.nan_to_num(Y / self.echelle_)

        xb = torch.tensor(X)
        yb = torch.tensor(Yn)
        mb = torch.tensor(mask.astype(np.float32))
        # chaque cellule pèse 1, quel que soit son nombre de blocs observés
        wb = mb / mb.sum(dim=1, keepdim=True).clamp(min=1.0)

        self.net_ = self._build(n_features=X.shape[2])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net_(xb)
            perte = (wb * (pred - yb) ** 2).sum() / len(curves)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
            opt.step()
            sched.step()
        self.net_.eval()
        self.perte_ = float(perte.detach())

    def _trajectory(self, T, C):
        torch = _torch()
        with torch.no_grad():
            pred = self.net_(torch.tensor(self._inputs(T, C)[None, ...]))
        perte = pred.numpy().ravel() * self.echelle_
        grid = self._grid()
        return self.a_ - np.interp(CYCLES, grid, perte)

    def _build(self, n_features):
        raise NotImplementedError

    # torch se pickle mal en présence de tenseurs non contigus : on ne garde
    # que les poids, et on reconstruit le réseau au chargement.
    def __getstate__(self):
        etat = dict(self.__dict__)
        net = etat.pop("net_", None)
        if net is not None:
            etat["_poids"] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        return etat

    def __setstate__(self, etat):
        poids = etat.pop("_poids", None)
        self.__dict__.update(etat)
        if poids is not None:
            torch = _torch()
            self.net_ = self._build(n_features=self._n_features)
            self.net_.load_state_dict({k: torch.tensor(v) for k, v in poids.items()})
            self.net_.eval()


class LstmModel(_SequenceCandidate):
    """LSTM sur la séquence de cycles, sortie = incréments de perte cumulés."""

    name = "lstm"

    def _build(self, n_features):
        torch = _torch()
        self._n_features = n_features
        hidden, autoreg = self.hidden, self.autoregressive

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(n_features + (1 if autoreg else 0),
                                          hidden, batch_first=True)
                self.head = torch.nn.Linear(hidden, 1)

            def forward(self, x):
                if not autoreg:
                    h, _ = self.lstm(x)
                    d = torch.nn.functional.softplus(self.head(h)).squeeze(-1)
                    return torch.cumsum(d, dim=1)
                B, L, _ = x.shape
                etat, cumul, sorties = None, torch.zeros(B, 1), []
                for t in range(L):
                    pas = torch.cat([x[:, t:t + 1, :], cumul[:, None, :]], dim=-1)
                    h, etat = self.lstm(pas, etat)
                    d = torch.nn.functional.softplus(self.head(h)).squeeze(-1)
                    cumul = cumul + d
                    sorties.append(cumul)
                return torch.cat(sorties, dim=1)

        return Net()


class TransformerModel(_SequenceCandidate):
    """Transformer encodeur sur la séquence de cycles, même tête cumulative."""

    name = "transformer"

    def __init__(self, n_steps=240, hidden=64, epochs=5000, lr=0.003,
                 weight_decay=1e-4, n_layers=3, n_heads=4, autoregressive=False):
        super().__init__(n_steps=n_steps, hidden=hidden, epochs=epochs, lr=lr,
                         weight_decay=weight_decay, autoregressive=False)
        self.n_layers = n_layers
        self.n_heads = n_heads

    def _build(self, n_features):
        torch = _torch()
        self._n_features = n_features
        hidden, n_layers, n_heads, n_steps = self.hidden, self.n_layers, self.n_heads, self.n_steps

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(n_features, hidden)
                self.pos = torch.nn.Parameter(torch.zeros(1, n_steps, hidden))
                torch.nn.init.normal_(self.pos, std=0.02)
                couche = torch.nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
                    dropout=0.0, batch_first=True, norm_first=True)
                self.enc = torch.nn.TransformerEncoder(couche, num_layers=n_layers)
                self.head = torch.nn.Linear(hidden, 1)

            def forward(self, x):
                h = self.enc(self.proj(x) + self.pos)
                d = torch.nn.functional.softplus(self.head(h)).squeeze(-1)
                return torch.cumsum(d, dim=1)

        return Net()
