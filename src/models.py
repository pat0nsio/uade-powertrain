"""Ensamble LightGBM + RSF + GRU + autoencoder con stacking; 20 % de vehículos en holdout, 5 folds OOF."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.ensemble import RandomSurvivalForest

from src.config import CFG, device
from src.features import CATS, HORIZONS, MAXS, MINS, MSG_W, SUMS, TRIP_W, feature_cols

SEED, K = 0, 5
MOD = Path("models")
GBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=SEED)
if (MOD / "gbm_params.json").exists():  # elegidos por `python -m src.models tune` (solo con folds de train)
    GBM_PARAMS.update(json.load(open(MOD / "gbm_params.json"))["best"])
GBM_PARAMS["device_type"] = CFG["lightgbm"]["device"]  # solo entrenamiento; la predicción de LightGBM es en CPU
G = CFG["gru"]
torch.manual_seed(SEED)
if CFG["torch"]["threads"]:
    torch.set_num_threads(CFG["torch"]["threads"])
DEV = device()  # config.local.json: "auto" / "cpu" / "cuda" (ROCm también se expone como "cuda")
AMP = CFG["torch"]["amp"] and DEV.type == "cuda"  # fp16 solo en GPU


def autocast():
    return torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP)


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


# ---------------- 3. Red secuencial: GRU con atención (+ features tabulares, + riesgo discreto) ----------------
DAILY = SUMS + TRIP_W + MSG_W + MAXS + MINS
WEEKS = 26  # cabeza de riesgo discreto: hazard semanal hasta 26 semanas
# elegida con `python -m src.models nn` (OOF); head: "bce" o "hazard" semanal con censura
ARCH = dict(seq=180, tab=True, head="hazard", pre=False)
F = torch.nn.functional


class SeqNet(torch.nn.Module):
    def __init__(self, n_in, n_tab=0, n_cats=(), n_out=len(HORIZONS), h=64):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.GELU())
        self.gru = torch.nn.GRU(h, h, batch_first=True, num_layers=1)
        self.att = torch.nn.Linear(h, 1)
        self.embs = torch.nn.ModuleList(torch.nn.Embedding(n + 1, 4) for n in n_cats)
        n_side = n_tab + 4 * len(n_cats)
        self.side = torch.nn.Sequential(torch.nn.Linear(n_side, h), torch.nn.GELU()) if n_side else None
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear((3 if n_side else 2) * h, h),
                                        torch.nn.GELU(), torch.nn.Linear(h, n_out))

    def forward(self, x, pad, tab=None, cat=None):
        hs, _ = self.gru(self.inp(x))
        hs = hs.float()
        a = self.att(hs).float().squeeze(-1).masked_fill(pad, -1e9).softmax(1)  # float: -1e9 no entra en fp16
        z = [(a.unsqueeze(-1) * hs).sum(1), hs[:, -1]]
        if self.side is not None:
            z.append(self.side(torch.cat([tab] + [e(cat[:, j]) for j, e in enumerate(self.embs)], 1)).float())
        return self.head(torch.cat(z, 1)), a


def daily_matrix(cal):
    X = cal[DAILY].astype("float32").copy()
    X[SUMS] = np.log1p(X[SUMS].clip(lower=0))
    X["active"] = cal["active"].astype("float32")
    return X.values


class Seq:
    """Indexa ventanas de arch["seq"] días del calendario sin materializarlas (+ entradas tabulares por fila)."""

    def __init__(self, cal, f, arch=ARCH):
        self.arch, self.L, self.cal = arch, arch["seq"], cal
        self.raw = daily_matrix(cal)
        pos = cal.groupby("v").cumcount().values
        start = pd.Series(np.arange(len(cal)) - pos, index=cal.index)
        key = pd.MultiIndex.from_arrays([cal["v"], cal["day"]])
        where = pd.Series(np.arange(len(cal)), index=key)
        self.end = where.reindex(pd.MultiIndex.from_arrays([f["v"], f["day"]])).values
        self.start = start.values[self.end]
        self.tab_raw = self.cat = None
        if arch["tab"]:
            self.tab_raw = f[[c for c in feature_cols(f) if c not in CATS]].values.astype("float32")
            self.cat = torch.from_numpy(np.column_stack([f[c].cat.codes.values + 1 for c in CATS])).long().to(DEV)
            self.n_cats = [len(f[c].cat.categories) for c in CATS]

    def fit_scaler(self, rows):
        seg = self.raw[np.unique(self.end[rows])]
        self.mu, self.sd = np.nanmean(seg, 0), np.nanstd(seg, 0) + 1e-6
        self.X = np.nan_to_num((self.raw - self.mu) / self.sd).clip(-6, 6).astype("float32")
        self.X = np.vstack([self.X, np.zeros((1, self.X.shape[1]), "float32")])  # fila de padding
        self.Xt = torch.from_numpy(self.X).to(DEV)  # se sube una sola vez; los batches se arman en el dispositivo
        if self.tab_raw is not None:  # estandarizado con el train; NaN -> 0 + máscara
            t = self.tab_raw[rows]
            self.tmu, self.tsd = np.nanmean(t, 0), np.nan_to_num(np.nanstd(t, 0)) + 1e-6
            Z = np.nan_to_num((self.tab_raw - self.tmu) / self.tsd).clip(-6, 6)
            self.T = torch.from_numpy(np.hstack([Z, np.isnan(self.tab_raw)]).astype("float32")).to(DEV)

    def net(self):
        if self.tab_raw is None:
            return SeqNet(self.X.shape[1], n_out=WEEKS if self.arch["head"] == "hazard" else len(HORIZONS))
        return SeqNet(self.X.shape[1], self.T.shape[1], self.n_cats,
                      n_out=WEEKS if self.arch["head"] == "hazard" else len(HORIZONS))

    def batch(self, rows):
        idx = self.end[rows, None] - np.arange(self.L - 1, -1, -1)[None]
        pad = idx < self.start[rows, None]
        idx[pad] = len(self.X) - 1
        x, pad = self.Xt[torch.from_numpy(idx).to(DEV)], torch.from_numpy(pad).to(DEV)
        if self.tab_raw is None:
            return x, pad
        r = torch.from_numpy(rows).to(DEV)
        return x, pad, self.T[r], self.cat[r]


def hazard_target(f):
    """Semana del evento (si ocurre dentro de WEEKS semanas) o cantidad de semanas completas observadas sin evento."""
    tte, gap = f["tte"].values, f["gap_to_end"].values
    ev = ~np.isnan(tte) & (tte <= WEEKS * 7)
    k = np.where(ev, (np.nan_to_num(tte) - 1) // 7, np.minimum(gap // 7, WEEKS))
    return k.astype("int64"), ev.astype("float32")


def hazard_nll(logit, k, ev):
    """-log verosimilitud en tiempo discreto con censura: sum_{j<k} log(1-h_j) + ev * log(h_k)."""
    j = torch.arange(WEEKS, device=logit.device)
    ll = (-F.softplus(logit) * (j[None] < k[:, None])).sum(1)
    ll = ll + ev * -F.softplus(-logit.gather(1, k.clamp(max=WEEKS - 1)[:, None]).squeeze(1))
    return -ll.mean()


def hazard_out(logit):
    """P(evento <= H días) para cada horizonte (log-supervivencia interpolada dentro de la semana) y RUL (mediana)."""
    L = torch.cat([torch.zeros_like(logit[:, :1]), torch.cumsum(-F.softplus(logit), 1)], 1)  # log S en semanas 0..26
    out = []
    for d in HORIZONS:
        k, fr = int(d // 7), d / 7 - int(d // 7)
        out.append(1 - torch.exp(L[:, k] + fr * (L[:, k + 1] - L[:, k])))
    half = np.log(0.5)
    below = L < half
    i = below.float().argmax(1).clamp(min=1)
    a, b = L.gather(1, (i - 1)[:, None]).squeeze(1), L.gather(1, i[:, None]).squeeze(1)
    rul = 7 * ((i - 1) + (a - half) / (a - b).clamp(min=1e-6))
    out.append(torch.where(below.any(1), rul, torch.full_like(rul, WEEKS * 7)))  # si nunca cae: >= 26 semanas
    return torch.stack(out, 1)


def _ap(y, p, m):
    from sklearn.metrics import average_precision_score
    return np.mean([average_precision_score(y[m[:, j] > 0, j], p[m[:, j] > 0, j]) for j in range(y.shape[1])])


def fit_gru(seq, f, rows):
    """Ensamble de G["seeds"] redes con parada temprana por AP en ~15 % de los vehículos del train."""
    Y = f[[f"y{h}" for h in HORIZONS]].values.astype("float32")
    M = f[[f"m{h}" for h in HORIZONS]].values.astype("float32")
    hz = seq.arch["head"] == "hazard"
    veh, fail = f["v"].values[rows], f["failed"].values[rows]
    rng = np.random.default_rng(SEED)
    val_v = np.concatenate([rng.choice(u, max(1, int(round(G["val_frac"] * len(u)))), replace=False)
                            for u in (np.unique(veh[fail == 0]), np.unique(veh[fail == 1]))])
    inval = np.isin(veh, val_v)
    fit_rows, val_rows = rows[~inval], rows[inval]
    seq.fit_scaler(fit_rows)
    if seq.arch["pre"]:  # solo vehículos y días de este train (+ v1): nada del holdout, del fold de validación ni de T+
        from src.pretrain import pretrain
        seq.encoder = pretrain(seq, np.unique(f["v"].values[rows]), f["day"].values[rows].max())
    Yt, Mt = torch.from_numpy(Y).to(DEV), torch.from_numpy(M).to(DEV)
    Kt, Et = (torch.from_numpy(a).to(DEV) for a in hazard_target(f))
    pos_w = (torch.tensor((1 - Y[fit_rows].mean(0)) / Y[fit_rows].mean(0)).clamp(max=20) ** 0.5).to(DEV)
    anypos = Y[fit_rows].max(1) > 0
    nets = []
    for s in range(G["seeds"]):
        torch.manual_seed(SEED + s)
        rng = np.random.default_rng(SEED + s)
        net = seq.net().to(DEV)
        if seq.arch["pre"]:
            net.inp.load_state_dict(seq.encoder.inp.state_dict()); net.gru.load_state_dict(seq.encoder.gru.state_dict())
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
        scaler = torch.amp.GradScaler(DEV.type, enabled=AMP)
        best, best_ep, best_state = -1.0, 0, None
        for ep in range(G["max_epochs"]):
            net.train()
            # todos los positivos + 15% de negativos por época
            ep_rows = np.concatenate([fit_rows[anypos],
                                      rng.choice(fit_rows[~anypos], int(0.15 * (~anypos).sum()), replace=False)])
            rng.shuffle(ep_rows)
            for i in range(0, len(ep_rows), G["batch"]):
                b = ep_rows[i:i + G["batch"]]
                with autocast():
                    logit, _ = net(*seq.batch(b))
                bt = torch.from_numpy(b).to(DEV)
                if hz:
                    loss = hazard_nll(logit.float(), Kt[bt], Et[bt])
                else:
                    l = F.binary_cross_entropy_with_logits(logit.float(), Yt[bt], pos_weight=pos_w, reduction="none")
                    loss = (l * Mt[bt]).sum() / Mt[bt].sum().clamp(min=1)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            ap = _ap(Y[val_rows], pred_gru([net], seq, val_rows)[0][:, :len(HORIZONS)], M[val_rows])
            if ap > best:
                best, best_ep, best_state = ap, ep, {k: v.detach().clone() for k, v in net.state_dict().items()}
            elif ep - best_ep >= G["patience"]:
                break
        net.load_state_dict(best_state)
        print(f"    gru seed {s}: mejor época {best_ep} AP val {best:.4f}", flush=True)
        nets.append(net)
    return nets


@torch.no_grad()
def pred_gru(nets, seq, rows, return_att=False):
    """(media, desvío entre semillas[, atención]) de P(<=H) por horizonte + RUL."""
    ps, atts = [], []
    for net in nets:
        net.eval()
        p, a = [], []
        for i in range(0, len(rows), G["pred_batch"]):
            with autocast():
                logit, att = net(*seq.batch(rows[i:i + G["pred_batch"]]))
            logit = logit.float()
            if seq.arch["head"] == "hazard":
                p.append(hazard_out(logit).cpu().numpy())
            else:
                p.append(np.column_stack([torch.sigmoid(logit).cpu().numpy(), np.full(len(logit), np.nan)]))
            if return_att:
                a.append(att.cpu().numpy())
        ps.append(np.vstack(p))
        if return_att:
            atts.append(np.vstack(a))
    ps = np.stack(ps)
    out = (ps.mean(0), ps.std(0))
    return out + (np.mean(atts, 0),) if return_att else out


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
    return torch.from_numpy(Z).to(DEV), stats


def fit_ae(X):
    Z, stats = ae_prep(X)
    net = AE(Z.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(15):
        for b in torch.randperm(len(Z), generator=g).split(1024):
            b = b.to(DEV)
            loss = torch.nn.functional.smooth_l1_loss(net(Z[b]), Z[b])  # fp32: el AE es chico, no vale la pena AMP
            opt.zero_grad(); loss.backward(); opt.step()
    return net, stats


@torch.no_grad()
def pred_ae(model, X):
    net, stats = model
    Z, _ = ae_prep(X, stats)
    return torch.cat([torch.nn.functional.smooth_l1_loss(net(z), z, reduction="none").mean(1)
                      for z in Z.split(65536)]).cpu().numpy()


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
    for c in ([f"{m}{h}" for m in ("gbm", "gru") for h in HORIZONS] + [f"gru{h}_std" for h in HORIZONS]
              + [f"rsf{t}" for t in SURV_T] + ["rul", "gru_rul", "ae_err"]):
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
        nets = fit_gru(seq, f, tr[f["usable"].values[tr]])
        for rows, w in tgt:
            p, sd = pred_gru(nets, seq, rows)
            for j, h in enumerate(HORIZONS):
                P.iloc[rows, P.columns.get_loc(f"gru{h}")] += w * p[:, j]
                P.iloc[rows, P.columns.get_loc(f"gru{h}_std")] += w * sd[:, j]
            P.iloc[rows, P.columns.get_loc("gru_rul")] += w * p[:, -1]
        del nets
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
    nets = fit_gru(seq, f, trall[f["usable"].values[trall]])
    torch.save({"arch": ARCH, "nets": [n.state_dict() for n in nets]}, MOD / "gru.pt")
    if ARCH["pre"]:
        torch.save(seq.encoder.state_dict(), MOD / "encoder.pt")
    joblib.dump((seq.mu, seq.sd), MOD / "gru_scaler.joblib")
    *_, att = pred_gru(nets, seq, np.arange(len(f)), return_att=True)
    np.save(MOD / "gru_attention.npy", att.astype("float16"))

    out = pd.concat([f[["v", "day", "fold", "failed", "tte", "gap_to_end", "usable"]
                       + [f"{p}{h}" for p in ("y", "m") for h in HORIZONS]], P], axis=1)
    out.to_parquet("data/preds.parquet")
    print("listo:", len(out), "filas")


# Grilla chica y explícita (regularización creciente); la selección usa OOF de los folds de train, nunca el holdout
GRID = [
    dict(),  # configuración original
    dict(num_leaves=15, min_child_samples=500, n_estimators=600),
    dict(num_leaves=15, min_child_samples=1000, reg_lambda=10.0, n_estimators=600),
    dict(num_leaves=31, min_child_samples=1000, reg_lambda=10.0, n_estimators=400),
    dict(num_leaves=7, min_child_samples=500, n_estimators=800),
    dict(num_leaves=31, min_child_samples=500, colsample_bytree=0.3, reg_lambda=5.0),
    dict(num_leaves=15, min_child_samples=500, extra_trees=True, n_estimators=800),
    dict(num_leaves=15, min_child_samples=1000, colsample_bytree=0.3, reg_lambda=10.0, learning_rate=0.02,
         n_estimators=900),
]


def tune(h=90):
    from sklearn.metrics import average_precision_score, roc_auc_score
    f = pd.read_parquet("data/features.parquet")
    cols = feature_cols(f)
    f["fold"] = split(f)
    base = {k: v for k, v in GBM_PARAMS.items()}
    for k in {k for g in GRID for k in g}:  # partir siempre de la configuración original
        base.pop(k, None)
    base.update(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, colsample_bytree=0.5,
                reg_lambda=1.0, extra_trees=False)
    res = []
    for i, g in enumerate(GRID):
        params = {**base, **g}
        oof = pd.Series(np.nan, index=f.index)
        fit_auc = []
        for k in range(K):
            tr = (f["fold"] >= 0) & (f["fold"] != k) & f[f"m{h}"]
            va = (f["fold"] == k) & f[f"m{h}"]
            m = lgb.LGBMClassifier(**params).fit(f.loc[tr, cols], f.loc[tr, f"y{h}"])
            oof[va] = m.predict_proba(f.loc[va, cols])[:, 1]
            fit_auc.append(roc_auc_score(f.loc[tr, f"y{h}"], m.predict_proba(f.loc[tr, cols])[:, 1]))
        ok = oof.notna()
        r = dict(config=i, **g, train_auc=np.mean(fit_auc), oof_auc=roc_auc_score(f.loc[ok, f"y{h}"], oof[ok]),
                 oof_ap=average_precision_score(f.loc[ok, f"y{h}"], oof[ok]))
        r["gap"] = r["train_auc"] - r["oof_auc"]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        res.append(r)
    res = pd.DataFrame(res)
    # criterio: mejor promedio de rankings de AUC y AP OOF
    best = int((res["oof_auc"].rank() + res["oof_ap"].rank()).idxmax())
    MOD.mkdir(exist_ok=True)
    json.dump({"best": {**base, **GRID[best]}, "horizon": h, "results": res.to_dict("records")},
              open(MOD / "gbm_params.json", "w"), indent=2, default=float)
    print("elegida:", best, GRID[best])


def nn_cv(tag, **arch):
    """Solo la red: AUC OOF con IC y peso en el stacking -> data/nn_{tag}.parquet (holdout solo de referencia)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.evaluate import cluster_ci
    arch = {**ARCH, **arch}
    print(f"== nn {tag}: {arch} {G}", flush=True)
    f = pd.read_parquet("data/features.parquet")
    f["fold"] = split(f)
    seq = Seq(pd.read_parquet("data/calendar.parquet"), f, arch)
    test = np.where(f["fold"] == -1)[0]
    out, sd = np.zeros((len(f), len(HORIZONS) + 1)), np.zeros((len(f), len(HORIZONS) + 1))
    for k in range(K):
        print(f"fold {k}", flush=True)
        tr, va = np.where((f["fold"] != k) & (f["fold"] != -1))[0], np.where(f["fold"] == k)[0]
        nets = fit_gru(seq, f, tr[f["usable"].values[tr]])
        for rows, w in ((va, 1.0), (test, 1.0 / K)):
            p, s = pred_gru(nets, seq, rows)
            out[rows] += w * p; sd[rows] += w * s
    cols = [f"gru{h}" for h in HORIZONS] + ["gru_rul"]
    P = pd.DataFrame(np.hstack([out, sd]), columns=cols + [c + "_std" for c in cols])
    P.to_parquet(f"data/nn_{tag}.parquet")
    base = pd.read_parquet("data/preds.parquet")
    res = {"tag": tag, **arch}
    for h in HORIZONS:
        m = f[f"m{h}"].values
        oof, te = m & (f["fold"].values >= 0), m & (f["fold"].values == -1)
        y, s, v = f[f"y{h}"].values, P[f"gru{h}"].values, f["v"].values
        ci, _ = cluster_ci(y[oof], s[oof], v[oof], n=200)
        B = base.assign(**{f"gru{h}": s})
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(meta_X(B[oof], h), y[oof])
        res[f"H{h}"] = dict(oof_auc=roc_auc_score(y[oof], s[oof]), oof_ci=ci,
                            oof_ap=average_precision_score(y[oof], s[oof]), holdout_auc=roc_auc_score(y[te], s[te]),
                            stack_w=meta.coef_[0].round(3).tolist(),
                            stack_holdout_auc=roc_auc_score(y[te], meta.predict_proba(meta_X(B[te], h))[:, 1]))
        r = res[f"H{h}"]
        print(f"  H{h}: OOF AUC {r['oof_auc']:.3f} {np.round(ci, 3)} AP {r['oof_ap']:.3f}"
              f" | holdout {r['holdout_auc']:.3f}"
              f" | stack (gbm, gru, rsf, ae) {r['stack_w']} holdout {r['stack_holdout_auc']:.3f}", flush=True)
    with open("data/nn_results.jsonl", "a") as fh:
        fh.write(json.dumps(res, default=float) + "\n")


if __name__ == "__main__":
    import sys
    if "tune" in sys.argv:
        tune()
    elif "nn" in sys.argv:  # python -m src.models nn <tag> [seq=180 tab=1 head=hazard]
        kv = dict(a.split("=") for a in sys.argv[3:])
        nn_cv(sys.argv[2], **{k: (v if k == "head" else int(v)) for k, v in kv.items()})
    elif "whatif" in sys.argv:
        fit_whatif()
    else:
        main()
        fit_whatif()
