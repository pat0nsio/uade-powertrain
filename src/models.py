"""Ensamble de 4 enfoques + stacking, con split por vehículo.

  1. LightGBM (features de ventana)          -> p(evento en H días), H = 30/60/90
  2. Random Survival Forest (landmark)        -> curva de supervivencia, RUL
  3. GRU multi-tarea con atención (PyTorch)   -> secuencia diaria de 60 días -> p(H)
  4. Autoencoder entrenado solo con sanos     -> error de reconstrucción = Health Index
  5. Stacking: regresión logística por horizonte sobre las predicciones OOF

Protocolo: 20% de vehículos en holdout (nunca vistos). Sobre el 80% restante, 5 folds agrupados por vehículo
producen predicciones OOF (entrenan el stacker); el holdout recibe el promedio de los 5 modelos.
Salida: data/preds.parquet, models/*.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.ensemble import RandomSurvivalForest

from src.features import HORIZONS, MAXS, MINS, MSG_W, SUMS, TRIP_W, feature_cols

SEED, K, SEQ = 0, 5, 60
MOD = Path("models")
GBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))


def split(f):
    veh = f.groupby("v")["failed"].first()
    rng = np.random.default_rng(SEED)
    test = set()
    for lab in (0, 1):
        vs = veh[veh == lab].index.values
        test |= set(rng.choice(vs, int(round(len(vs) * 0.2)), replace=False))
    fold = pd.Series(-1, index=veh.index)
    tr = veh[~veh.index.isin(test)]
    for k, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(tr, tr, tr.index)):
        fold[tr.index[te]] = k
    return f["v"].map(fold).values


# ---------------- 1. LightGBM ----------------
def fit_gbm(X, y):
    return lgb.LGBMClassifier(**GBM_PARAMS).fit(X, y)


# ---------------- 2. Random Survival Forest ----------------
SURV_T = (30, 60, 90, 180)


def surv_target(d):
    ev = d["tte"].notna()
    t = np.where(ev, d["tte"], d["gap_to_end"]).astype(float)
    return np.array(list(zip(ev, t)), dtype=[("e", bool), ("t", float)])


def fit_rsf(X, d):
    ok = (d["usable"] & ((d["tte"].notna()) | (d["gap_to_end"] > 0))).values
    # landmark: una observación cada ~14 días activos por vehículo (evita miles de filas casi idénticas)
    ok = ok & (d.groupby("v").cumcount() % 14 == 0).values
    return RandomSurvivalForest(n_estimators=150, min_samples_leaf=40, max_features="sqrt", n_jobs=-1,
                                random_state=SEED).fit(X[ok], surv_target(d[ok]))


def pred_rsf(m, X):
    out = []
    for i in range(0, len(X), 20000):
        S = m.predict_survival_function(X[i:i + 20000], return_array=True)
        times = m.unique_times_
        cols = [1 - S[:, min(np.searchsorted(times, t), len(times) - 1)] for t in SURV_T]
        # RUL = primer t con S(t) < 0.5 (si nunca cae, se reporta el horizonte máximo observado)
        below = S < 0.5
        rul = np.where(below.any(1), times[below.argmax(1)], times[-1])
        out.append(np.column_stack(cols + [rul]))
    return np.vstack(out)


# ---------------- 3. GRU multi-tarea con atención ----------------
DAILY = SUMS + TRIP_W + MSG_W + MAXS + MINS


class SeqNet(torch.nn.Module):
    def __init__(self, n_in, h=64):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.GELU())
        self.gru = torch.nn.GRU(h, h, batch_first=True, num_layers=1)
        self.att = torch.nn.Linear(h, 1)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2 * h, h), torch.nn.GELU(),
                                        torch.nn.Linear(h, len(HORIZONS)))

    def forward(self, x, pad):
        hs, _ = self.gru(self.inp(x))
        a = self.att(hs).squeeze(-1).masked_fill(pad, -1e9).softmax(1)
        ctx = (a.unsqueeze(-1) * hs).sum(1)
        return self.head(torch.cat([ctx, hs[:, -1]], 1)), a


class Seq:
    """Indexa ventanas de 90 días del calendario sin materializarlas."""

    def __init__(self, cal, f):
        X = cal[DAILY].astype("float32").copy()
        X[SUMS] = np.log1p(X[SUMS].clip(lower=0))
        X["active"] = cal["active"].astype("float32")
        self.raw = X.values
        pos = cal.groupby("v").cumcount().values
        start = pd.Series(np.arange(len(cal)) - pos, index=cal.index)
        key = pd.MultiIndex.from_arrays([cal["v"], cal["day"]])
        where = pd.Series(np.arange(len(cal)), index=key)
        self.end = where.reindex(pd.MultiIndex.from_arrays([f["v"], f["day"]])).values
        self.start = start.values[self.end]

    def fit_scaler(self, rows):
        seg = self.raw[np.unique(self.end[rows])]
        self.mu, self.sd = np.nanmean(seg, 0), np.nanstd(seg, 0) + 1e-6
        self.X = np.nan_to_num((self.raw - self.mu) / self.sd).clip(-6, 6).astype("float32")
        self.X = np.vstack([self.X, np.zeros((1, self.X.shape[1]), "float32")])  # fila de padding

    def batch(self, rows):
        idx = self.end[rows, None] - np.arange(SEQ - 1, -1, -1)[None]
        pad = idx < self.start[rows, None]
        idx[pad] = len(self.X) - 1
        return torch.from_numpy(self.X[idx]), torch.from_numpy(pad)


def fit_gru(seq, f, rows, epochs=6):
    Y = f[[f"y{h}" for h in HORIZONS]].values.astype("float32")
    M = f[[f"m{h}" for h in HORIZONS]].values.astype("float32")
    seq.fit_scaler(rows)
    net = SeqNet(seq.X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    pos_w = torch.tensor((1 - Y[rows].mean(0)) / Y[rows].mean(0)).clamp(max=20) ** 0.5
    rng = np.random.default_rng(SEED)
    anypos = Y[rows].max(1) > 0
    for ep in range(epochs):
        net.train()
        # todos los positivos + 15% de negativos por época
        ep_rows = np.concatenate([rows[anypos], rng.choice(rows[~anypos], int(0.15 * (~anypos).sum()), replace=False)])
        rng.shuffle(ep_rows)
        tot = 0
        for i in range(0, len(ep_rows), 512):
            b = ep_rows[i:i + 512]
            x, pad = seq.batch(b)
            logit, _ = net(x, pad)
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, torch.from_numpy(Y[b]), pos_weight=pos_w, reduction="none")
            loss = (l * torch.from_numpy(M[b])).sum() / torch.from_numpy(M[b]).sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        print(f"    gru epoch {ep} loss {tot / len(ep_rows):.4f}")
    return net


@torch.no_grad()
def pred_gru(net, seq, rows, return_att=False):
    net.eval()
    ps, atts = [], []
    for i in range(0, len(rows), 4096):
        x, pad = seq.batch(rows[i:i + 4096])
        logit, a = net(x, pad)
        ps.append(torch.sigmoid(logit).numpy()); atts.append(a.numpy())
    return (np.vstack(ps), np.vstack(atts)) if return_att else np.vstack(ps)


# ---------------- 4. Autoencoder (solo sanos) ----------------
class AE(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.GELU(), torch.nn.Linear(64, 12))
        self.dec = torch.nn.Sequential(torch.nn.Linear(12, 64), torch.nn.GELU(), torch.nn.Linear(64, n))

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_prep(X, stats=None):
    if stats is None:
        med = np.nanmedian(X, 0)
        q1, q3 = np.nanpercentile(X, [25, 75], 0)
        stats = (med, np.where(q3 - q1 > 0, q3 - q1, np.nanstd(X, 0) + 1e-6))
    Z = np.nan_to_num((X - stats[0]) / stats[1]).clip(-8, 8).astype("float32")
    return torch.from_numpy(Z), stats


def fit_ae(X):
    Z, stats = ae_prep(X)
    net = AE(Z.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(15):
        for b in torch.randperm(len(Z), generator=g).split(1024):
            loss = torch.nn.functional.smooth_l1_loss(net(Z[b]), Z[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net, stats


@torch.no_grad()
def pred_ae(model, X):
    net, stats = model
    Z, _ = ae_prep(X, stats)
    return torch.nn.functional.smooth_l1_loss(net(Z), Z, reduction="none").mean(1).numpy()


# ---------------- 5. Stacking ----------------
def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def meta_X(P, h):
    return np.column_stack([logit(P[f"gbm{h}"]), logit(P[f"gru{h}"]), logit(P[f"rsf{h}"]), np.log(P["ae_err"])])


# Restricciones monótonas físicas para el simulador what-if (el GBM libre no es causal: aprende sesgos de muestreo)
MONO_UP = ["sh_short5", "sh_short10", "sh_micro", "sh_urban", "sh_never_warm", "sh_cold_start", "sh_idle",
           "sh_trip_end_in_regen", "regen_stop_ratio"]
MONO_DOWN = ["km_per_trip", "speed"]


def monotone(cols):
    def sign(c):
        body = c.split("_", 1)[1] if c.startswith(("w7_", "w30_", "w90_", "life_")) else c
        if body in MONO_UP or c in ("days_since_regen", "km_since_regen"):
            return 1
        return -1 if body in MONO_DOWN else 0
    return [sign(c) for c in cols]


def fit_whatif():
    """GBM 90d con restricciones monótonas; se compara contra el GBM libre en el holdout."""
    from sklearn.metrics import roc_auc_score
    f = pd.read_parquet("data/features.parquet")
    p = pd.read_parquet("data/preds.parquet")
    cols = feature_cols(f)
    tr, te = (p["fold"] >= 0) & f["m90"], (p["fold"] == -1) & f["m90"]
    m = lgb.LGBMClassifier(**GBM_PARAMS, monotone_constraints=monotone(cols), monotone_constraints_method="advanced")
    m.fit(f.loc[tr, cols], f.loc[tr, "y90"])
    m.booster_.save_model(str(MOD / "gbm90_whatif.txt"))
    auc = roc_auc_score(f.loc[te, "y90"], m.predict_proba(f.loc[te, cols])[:, 1])
    print(f"what-if GBM monótono: {sum(x != 0 for x in monotone(cols))} features restringidas, "
          f"AUC holdout {auc:.3f} vs libre {roc_auc_score(f.loc[te, 'y90'], p.loc[te, 'gbm90']):.3f}")
    return auc


def main():
    MOD.mkdir(exist_ok=True)
    f = pd.read_parquet("data/features.parquet")
    cal = pd.read_parquet("data/calendar.parquet")
    cols = feature_cols(f)
    num_cols = [c for c in cols if f[c].dtype != "category"]
    f["fold"] = split(f)
    test = np.where(f["fold"] == -1)[0]
    P = pd.DataFrame(index=f.index)
    for c in [f"{m}{h}" for m in ("gbm", "gru") for h in HORIZONS] + [f"rsf{t}" for t in SURV_T] + ["rul", "ae_err"]:
        P[c] = 0.0
    seq = Seq(cal, f)
    Xn = f[num_cols].values.astype("float32")

    for k in range(K):
        print(f"fold {k}")
        tr, va = np.where((f["fold"] != k) & (f["fold"] != -1))[0], np.where(f["fold"] == k)[0]
        tgt = [(va, 1.0), (test, 1.0 / K)]
        for h in HORIZONS:
            r = tr[f[f"m{h}"].values[tr]]
            m = fit_gbm(f.loc[r, cols], f.loc[r, f"y{h}"])
            for rows, w in tgt:
                P.iloc[rows, P.columns.get_loc(f"gbm{h}")] += w * m.predict_proba(f.loc[rows, cols])[:, 1]
            if h == 90:
                top = pd.Series(m.booster_.feature_importance("gain"), cols).drop(
                    [c for c in cols if c not in num_cols]).nlargest(30).index
        print("  gbm ok")
        med = np.nanmedian(f.loc[tr, top].values, 0)
        Xr = lambda rows: np.where(np.isnan(f.loc[rows, top].values), med, f.loc[rows, top].values)
        rsf = fit_rsf(Xr(tr), f.iloc[tr])
        for rows, w in tgt:
            out = pred_rsf(rsf, Xr(rows))
            for j, t in enumerate(SURV_T):
                P.iloc[rows, P.columns.get_loc(f"rsf{t}")] += w * out[:, j]
            P.iloc[rows, P.columns.get_loc("rul")] += w * out[:, -1]
        del rsf
        print("  rsf ok")
        net = fit_gru(seq, f, tr[f["usable"].values[tr]])
        for rows, w in tgt:
            p = pred_gru(net, seq, rows)
            for j, h in enumerate(HORIZONS):
                P.iloc[rows, P.columns.get_loc(f"gru{h}")] += w * p[:, j]
        print("  gru ok")
        healthy_tr = tr[(f["failed"].values[tr] == 0)]
        ae = fit_ae(Xn[healthy_tr])
        for rows, w in tgt:
            P.iloc[rows, P.columns.get_loc("ae_err")] += w * pred_ae(ae, Xn[rows])
        print("  ae ok")

    oof = f["fold"].values >= 0
    for h in HORIZONS:
        m = oof & f[f"m{h}"].values
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(meta_X(P[m], h), f.loc[m, f"y{h}"])
        P[f"stack{h}"] = meta.predict_proba(meta_X(P, h))[:, 1]
        joblib.dump(meta, MOD / f"stack{h}.joblib")
        print(f"stack{h} coefs (gbm, gru, rsf, ae):", meta.coef_.round(3))

    # Health Index 0-100: percentil del error del AE respecto de días de vehículos sanos (OOF)
    ref = np.sort(P.loc[oof & (f["failed"].values == 0), "ae_err"].values)
    P["health"] = 100 * (1 - np.searchsorted(ref, P["ae_err"].values) / len(ref))

    # modelos finales sobre todo el train (para SHAP / what-if en el dashboard y la GRU con atención)
    trall = np.where(f["fold"] >= 0)[0]
    for h in HORIZONS:
        r = trall[f[f"m{h}"].values[trall]]
        fit_gbm(f.loc[r, cols], f.loc[r, f"y{h}"]).booster_.save_model(str(MOD / f"gbm{h}.txt"))
    net = fit_gru(seq, f, trall[f["usable"].values[trall]])
    torch.save(net.state_dict(), MOD / "gru.pt")
    joblib.dump((seq.mu, seq.sd), MOD / "gru_scaler.joblib")
    _, att = pred_gru(net, seq, np.arange(len(f)), return_att=True)
    np.save(MOD / "gru_attention.npy", att.astype("float16"))

    out = pd.concat([f[["v", "day", "fold", "failed", "tte", "gap_to_end", "usable"]
                       + [f"{p}{h}" for p in ("y", "m") for h in HORIZONS]], P], axis=1)
    out.to_parquet("data/preds.parquet")
    print("listo:", len(out), "filas")


if __name__ == "__main__":
    import sys
    fit_whatif() if "whatif" in sys.argv else (main(), fit_whatif())
