"""Pre-entrenamiento auto-supervisado del codificador de SeqNet (días enmascarados + semana siguiente)."""
import numpy as np
import pandas as pd
import torch

from src.features import calendar
from src.models import DEV, G, SEED, AMP, autocast, daily_matrix

NEXT = ["km", "n_regen", "soot_delta"]
F = torch.nn.functional


class Pre(torch.nn.Module):
    def __init__(self, n_in, h=64):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.GELU())  # mismos nombres que SeqNet
        self.gru = torch.nn.GRU(h, h, batch_first=True, num_layers=1)
        self.rec = torch.nn.Linear(h, n_in)
        self.nxt = torch.nn.Linear(h, len(NEXT))


def v1_calendar():
    d = pd.read_parquet("data/daily_v1.parquet")
    d["day"] = pd.to_datetime(d["day"])
    return calendar(d)


def pretrain(seq, vehicles, day_max):
    """Devuelve el codificador pre-entrenado (Pre). Usa el escalado ya ajustado en seq (fit_scaler)."""
    c = seq.cal
    c1 = v1_calendar()
    C = pd.concat([c[c["v"].isin(vehicles) & (c["day"] <= day_max)], c1[c1["day"] <= day_max]], ignore_index=True)
    X = np.nan_to_num((daily_matrix(C) - seq.mu) / seq.sd).clip(-6, 6).astype("float32")
    Xt = torch.from_numpy(np.vstack([X, np.zeros((1, X.shape[1]), "float32")])).to(DEV)
    n = len(C)
    pos = C.groupby("v", sort=False).cumcount().values
    start = np.arange(n) - pos
    last = start + C.groupby("v", sort=False)["v"].transform("size").values - 1
    # objetivo "semana siguiente": sumas de los días i+1..i+7 (cumsum global; solo se usa dentro del mismo vehículo)
    cs = np.vstack([np.zeros((1, len(NEXT))), np.cumsum(C[NEXT].fillna(0).values, 0)])
    ends = np.where(C["active"].values & (np.arange(n) + 7 <= last))[0]
    nxt = cs[ends + 8] - cs[ends + 1]
    nxt[:, 0] = np.log1p(nxt[:, 0].clip(min=0))
    nxt = (nxt - nxt.mean(0)) / (nxt.std(0) + 1e-6)
    Nt = torch.zeros((n, len(NEXT)), device=DEV)
    Nt[torch.from_numpy(ends).to(DEV)] = torch.from_numpy(nxt.astype("float32")).to(DEV)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pre = Pre(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(pre.parameters(), 2e-3, weight_decay=1e-3)
    scaler = torch.amp.GradScaler(DEV.type, enabled=AMP)
    L = seq.L
    print(f"    pre-entrenamiento: {C['v'].nunique()} vehículos, {len(ends)} ventanas", flush=True)
    for ep in range(G["pre_epochs"]):
        tot = 0.0
        for b in np.array_split(rng.permutation(ends), max(1, len(ends) // G["batch"])):
            idx = b[:, None] - np.arange(L - 1, -1, -1)[None]
            pad = torch.from_numpy(idx < start[b, None]).to(DEV)
            idx[idx < start[b, None]] = n
            x = Xt[torch.from_numpy(idx).to(DEV)]
            rate = torch.empty(len(b), 1, device=DEV).uniform_(0.15, 0.30)
            m = (torch.rand(len(b), L, device=DEV) < rate) & ~pad
            with autocast():
                hs, _ = pre.gru(pre.inp(x.masked_fill(m[..., None], 0.0)))
                rec, nx = pre.rec(hs[m]), pre.nxt(hs[:, -1])
            loss = F.mse_loss(rec.float(), x[m]) + F.mse_loss(nx.float(), Nt[torch.from_numpy(b).to(DEV)])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"    pre época {ep}: pérdida {tot / max(1, len(ends) // G['batch']):.4f}", flush=True)
    return pre
