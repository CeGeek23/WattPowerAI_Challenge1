# -*- coding: utf-8 -*-
"""Couplage LSTM -> Transformer sur la serie SOH reechantillonnee.

Le LSTM seul et le Transformer seul ont ete mesures puis ecartes : 4.82 et 5.48
de RMSE en LOCO contre 3.22 pour le couplage.

Trois leviers pour qu'un reseau tienne sur 6 sequences :
  - sortie = increments de perte cumules (softplus + cumsum) : monotonie garantie ;
  - mixup en (T, C) : conditions virtuelles interpolees entre deux cellules.
    Desactive par defaut : mesure a 6.00 de RMSE en LOCO contre 4.11 sans lui.
    L'interpolation lineaire des cibles est fausse ici, deux courbes de fade ne
    s'interpolent pas en amplitude mais en echelle de temps ;
  - pre-entrainement optionnel sur les courbes publiques, puis fine-tuning.
"""
import numpy as np

from .base import CYCLES, N_CYCLES, SEED, Candidate, cond_features, cycle_features

# Le pre-entrainement ne depend pas des cellules cibles : on le fait une fois et
# on reutilise les poids (sinon la LOCO le refait a chaque pli).
_CACHE_PRETRAIN = {}


def _torch():
    import torch
    return torch


def appareil(choix="auto"):
    """cuda si dispo, sinon mps (Apple), sinon cpu. `choix` force un appareil."""
    torch = _torch()
    if choix != "auto":
        return torch.device(choix)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _SequenceCandidate(Candidate):
    """Grille, cible, entrainement, prediction : commun aux trois architectures."""

    name = "sequence"
    needs = ("torch",)

    def __init__(self, n_steps=240, hidden=64, epochs=4000, lr=0.005, weight_decay=1e-4,
                 n_mixup=0, pretrain_curves=None, pretrain_epochs=3000, device="auto"):
        self.n_steps = n_steps
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_mixup = n_mixup
        self.pretrain_curves = pretrain_curves      # [(T, C, n, soh)] publiques
        self.pretrain_epochs = pretrain_epochs
        self.device = device

    # ------------------------------------------------------------- donnees
    def _grid(self):
        edges = np.linspace(0, N_CYCLES, self.n_steps + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def _inputs(self, T, C):
        grid = self._grid()
        cond = np.repeat(cond_features([T], [C]), len(grid), axis=0)
        return np.column_stack([cycle_features(grid), cond]).astype(np.float32)

    def _targets(self, curves, a_ref):
        """Perte de SOH moyenne par bloc, NaN la ou rien n'est observe."""
        bloc = np.clip((np.arange(N_CYCLES) * self.n_steps) // N_CYCLES, 0, self.n_steps - 1)
        Y = np.full((len(curves), self.n_steps), np.nan, dtype=np.float32)
        for k, (_, _, n, y) in enumerate(curves):
            b = bloc[np.clip(n.astype(int), 1, N_CYCLES) - 1]
            somme = np.bincount(b, weights=a_ref - y, minlength=self.n_steps)
            compte = np.bincount(b, minlength=self.n_steps)
            vus = compte > 0
            Y[k, vus] = (somme[vus] / compte[vus]).astype(np.float32)
        return Y

    def _tenseurs(self, curves, a_ref):
        torch = _torch()
        d = self.appareil_
        X = np.stack([self._inputs(T, C) for T, C, _, _ in curves])
        Y = self._targets(curves, a_ref)
        M = np.isfinite(Y).astype(np.float32)
        return (torch.tensor(X, device=d), torch.tensor(np.nan_to_num(Y), device=d),
                torch.tensor(M, device=d),
                np.array([[T, C] for T, C, _, _ in curves], dtype=float))

    def _mixup(self, xb, yb, mb, conds, rng):
        """Conditions virtuelles interpolees : la reponse en (T, C) doit etre continue."""
        torch = _torch()
        if self.n_mixup <= 0 or len(conds) < 2:
            return xb, yb, mb
        i = rng.integers(0, len(conds), self.n_mixup)
        j = rng.integers(0, len(conds), self.n_mixup)
        garde = i != j
        i, j = i[garde], j[garde]
        if not len(i):
            return xb, yb, mb
        lam_np = rng.uniform(0.2, 0.8, len(i)).astype(np.float32)
        melange = np.stack([self._inputs(*(lam_np[k] * conds[i[k]]
                                           + (1 - lam_np[k]) * conds[j[k]]))
                            for k in range(len(i))])
        lam = torch.tensor(lam_np, device=self.appareil_)[:, None]
        return (torch.cat([xb, torch.tensor(melange, device=self.appareil_)]),
                torch.cat([yb, lam * yb[i] + (1 - lam) * yb[j]]),
                torch.cat([mb, mb[i] * mb[j]]))

    # -------------------------------------------------------- entrainement
    def _boucle(self, xb, yb, mb, conds, epochs, lr, rng):
        torch = _torch()
        xa, ya = xb, yb
        pa = mb / mb.sum(dim=1, keepdim=True).clamp(min=1.0)
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        self.net_.train()
        perte = torch.tensor(0.0)
        histo = []
        for ep in range(epochs):
            if self.n_mixup and ep % 25 == 0:      # nouveau lot virtuel de temps en temps
                xa, ya, ma = self._mixup(xb, yb, mb, conds, rng)
                pa = ma / ma.sum(dim=1, keepdim=True).clamp(min=1.0)
            opt.zero_grad()
            perte = (pa * (self.net_(xa) - ya) ** 2).sum() / len(xa)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
            opt.step()
            sched.step()
            if ep % 25 == 0:
                histo.append((ep, float(perte.detach())))
        self.net_.eval()
        self.historique_ = histo          # (epoque, perte) pour tracer la convergence
        return float(perte.detach())

    def _fit(self, curves):
        torch = _torch()
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        self.appareil_ = appareil(self.device)

        xb, yb, mb, conds = self._tenseurs(curves, self.a_)
        self.echelle_ = float(max(np.nanmax(yb.cpu().numpy()), 1.0))
        yb = yb / self.echelle_
        self.net_ = self._build(n_features=xb.shape[2]).to(self.appareil_)
        self.n_pretrain_ = 0

        if self.pretrain_curves:
            pc = [c for c in self.pretrain_curves if len(c[2]) >= 20]
            if len(pc) >= 3:
                cle = (type(self).__name__, self.hidden, self.n_steps, self.pretrain_epochs,
                       len(pc), round(self.echelle_, 3))
                if cle in _CACHE_PRETRAIN:
                    self.net_.load_state_dict(
                        {k: torch.tensor(v, device=self.appareil_)
                         for k, v in _CACHE_PRETRAIN[cle].items()})
                else:
                    a_pub = float(np.mean([np.percentile(y, 99.5) for *_, y in pc]))
                    xp, yp, mp, cp = self._tenseurs(pc, a_pub)
                    self.perte_pretrain_ = self._boucle(xp, yp / self.echelle_, mp, cp,
                                                        self.pretrain_epochs, self.lr, rng)
                    self.histo_pretrain_ = self.historique_
                    _CACHE_PRETRAIN[cle] = {k: v.cpu().numpy()
                                            for k, v in self.net_.state_dict().items()}
                self.n_pretrain_ = len(pc)
        # apres pre-entrainement on reprend plus doucement : c'est du fine-tuning
        self.perte_ = self._boucle(xb, yb, mb, conds, self.epochs,
                                   self.lr * (0.3 if self.n_pretrain_ else 1.0), rng)

    def _trajectory(self, T, C):
        torch = _torch()
        d = getattr(self, "appareil_", None) or appareil(self.device)
        with torch.no_grad():
            pred = self.net_(torch.tensor(self._inputs(T, C)[None, ...], device=d))
        perte = pred.detach().cpu().numpy().ravel() * self.echelle_
        return self.a_ - np.interp(CYCLES, self._grid(), perte)

    def _build(self, n_features):
        raise NotImplementedError

    def __getstate__(self):
        etat = dict(self.__dict__)
        etat.pop("appareil_", None)          # un torch.device ne se pickle pas proprement
        net = etat.pop("net_", None)
        if net is not None:
            etat["_poids"] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        return etat

    def __setstate__(self, etat):
        poids = etat.pop("_poids", None)
        self.__dict__.update(etat)
        if poids is not None:
            torch = _torch()
            self.appareil_ = appareil(self.device)
            self.net_ = self._build(n_features=self._n_features).to(self.appareil_)
            self.net_.load_state_dict({k: torch.tensor(v, device=self.appareil_)
                                       for k, v in poids.items()})
            self.net_.eval()


class LstmTransformerModel(_SequenceCandidate):
    """Couplage empile : le LSTM code la dynamique locale, l'attention l'integre."""

    name = "lstm+transformer"

    def __init__(self, n_layers=2, n_heads=4, lr=0.003, **kw):
        super().__init__(lr=lr, **kw)
        self.n_layers = n_layers
        self.n_heads = n_heads

    def _build(self, n_features):
        torch = _torch()
        self._n_features = n_features
        hidden, n_layers, n_heads, n_steps = self.hidden, self.n_layers, self.n_heads, self.n_steps

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(n_features, hidden, batch_first=True)
                self.pos = torch.nn.Parameter(torch.zeros(1, n_steps, hidden))
                torch.nn.init.normal_(self.pos, std=0.02)
                couche = torch.nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
                    dropout=0.0, batch_first=True, norm_first=True)
                self.enc = torch.nn.TransformerEncoder(couche, num_layers=n_layers)
                self.head = torch.nn.Linear(hidden, 1)

            def forward(self, x):
                h, _ = self.lstm(x)
                h = self.enc(h + self.pos)
                d = torch.nn.functional.softplus(self.head(h)).squeeze(-1)
                return torch.cumsum(d, dim=1)

        return Net()
