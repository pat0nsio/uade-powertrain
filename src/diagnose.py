"""Diagnóstico: ¿el techo de desempeño lo ponen los datos?

  learning  Curva de aprendizaje: LightGBM y la red con 25/50/75/100 % de los vehículos de train, submuestreados por
            vehículo (estratificado por fallado) dentro de cada fold; AUC/AP OOF sobre los folds de validación completos.
  temporal  Descompone la caída del holdout por vehículo (~0.78) a la validación temporal (~0.64), con el LightGBM y los
            vehículos del holdout: período de entrenamiento (todo / etiquetas conocidas en T) x período evaluado
            (< T / >= T), más un control con el train completo submuestreado al tamaño y positivos del de "antes de T".

Salidas: data/learning_curve.json, data/temporal_decomp.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import cluster_ci
from src.features import HORIZONS, feature_cols
from src.models import K, fit_gbm, split
from src.temporal import T, known_at


def load():
    f = pd.read_parquet("data/features.parquet")
    f["day"] = pd.to_datetime(f["day"])
    f["fold"] = split(f)
    return f


def paired_ci(y, a, b, v, n=300, seed=0):
    """IC95 de AUC(b) - AUC(a) remuestreando vehículos (mismas filas para los dos modelos)."""
    rng = np.random.default_rng(seed)
    g = list(pd.Series(np.arange(len(y))).groupby(v).indices.values())
    d = []
    for _ in range(n):
        i = np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))])
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).round(4).tolist()


# ---------------- 1. curva de aprendizaje ----------------
def learning(fracs=(0.25, 0.5, 0.75), reps_gbm=5, reps_nn=2):
    f = load()
    cols = feature_cols(f)
    failed = f.groupby("v")["failed"].first()
    oof = f["fold"].values >= 0
    folds = [(np.where((f["fold"] >= 0) & (f["fold"] != k))[0], np.where(f["fold"] == k)[0]) for k in range(K)]

    def subsample(tr, frac, rng):
        vs = np.unique(f["v"].values[tr])
        keep = np.concatenate([rng.choice(g, max(1, round(frac * len(g))), replace=False)
                               for g in (vs[failed[vs].values == 0], vs[failed[vs].values == 1])])
        return tr[np.isin(f["v"].values[tr], keep)], len(keep), int(failed[keep].sum())

    def report(model, frac, rep, P, n_veh, n_fail):
        r = {"model": model, "frac": frac, "rep": rep, "train_vehicles_per_fold": n_veh, "train_failed_per_fold": n_fail}
        for j, h in enumerate(HORIZONS):
            m = oof & f[f"m{h}"].values
            y, s = f[f"y{h}"].values[m], P[m, j]
            r[f"H{h}"] = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                          "auc_ci95": cluster_ci(y, s, f["v"].values[m], n=100)[0]}
        print(f"  {model} {frac:.0%} rep {rep}: vehículos/fold {n_veh:.0f} (fallados {n_fail:.0f}) | "
              + " ".join(f"H{h} {r[f'H{h}']['auc']:.3f}" for h in HORIZONS), flush=True)
        return r

    res = []
    # LightGBM
    for frac in fracs + (1.0,):
        for rep in range(1 if frac == 1 else reps_gbm):
            rng = np.random.default_rng(1000 * rep + int(100 * frac))
            P, nv, nf = np.full((len(f), len(HORIZONS)), np.nan), [], []
            for tr, va in folds:
                trs, a, b = subsample(tr, frac, rng) if frac < 1 else (tr, f["v"].iloc[tr].nunique(),
                                                                         int(failed[f["v"].iloc[tr].unique()].sum()))
                nv.append(a); nf.append(b)
                for j, h in enumerate(HORIZONS):
                    r = trs[f[f"m{h}"].values[trs]]
                    P[va, j] = fit_gbm(f.loc[r, cols], f.loc[r, f"y{h}"]).predict_proba(f.loc[va, cols])[:, 1]
            res.append(report("gbm", frac, rep, P, np.mean(nv), np.mean(nf)))
    # red (ARCH vigente); al 100 % se usan las predicciones OOF del pipeline final (mismos folds y arquitectura)
    from src.models import Seq, fit_gru, pred_gru
    seq = Seq(pd.read_parquet("data/calendar.parquet"), f)
    usable = f["usable"].values
    for frac in fracs:
        for rep in range(reps_nn):
            rng = np.random.default_rng(1000 * rep + int(100 * frac))
            P, nv, nf = np.full((len(f), len(HORIZONS)), np.nan), [], []
            for tr, va in folds:
                trs, a, b = subsample(tr, frac, rng)
                nv.append(a); nf.append(b)
                P[va] = pred_gru(fit_gru(seq, f, trs[usable[trs]]), seq, va)[0][:, :len(HORIZONS)]
            res.append(report("red", frac, rep, P, np.mean(nv), np.mean(nf)))
    base = pd.read_parquet("data/preds.parquet")
    assert (base["v"].values == f["v"].values).all() and (base["fold"].values == f["fold"].values).all()
    res.append(report("red", 1.0, 0, base[[f"gru{h}" for h in HORIZONS]].values,
                      np.mean([f["v"].iloc[tr].nunique() for tr, _ in folds]),
                      np.mean([failed[f["v"].iloc[tr].unique()].sum() for tr, _ in folds])))
    json.dump(res, open("data/learning_curve.json", "w"), indent=2, default=float)
    summary = pd.DataFrame([{"model": r["model"], "frac": r["frac"], **{f"H{h}": r[f"H{h}"]["auc"] for h in HORIZONS}}
                            for r in res]).groupby(["model", "frac"]).agg(["mean", "std", "count"]).round(4)
    print(summary.to_string())


# ---------------- 2. descomposición de la caída temporal ----------------
def temporal_decomp(reps=3):
    f = load()
    cols = feature_cols(f)
    trv, tev = (f["fold"] >= 0).values, (f["fold"] == -1).values
    post = (f["day"] >= T).values
    res = {"T": str(T.date())}
    for h in HORIZONS:
        m, y = f[f"m{h}"].values, f[f"y{h}"].values
        train = {"todo": trv & m, "antes_de_T": trv & known_at(f, h, T).values}
        n_pre, pos_pre = int(train["antes_de_T"].sum()), int(y[train["antes_de_T"]].sum())
        preds = {}
        for name, rows in train.items():
            preds[name] = fit_gbm(f.loc[rows, cols], y[rows]).predict_proba(f[cols])[:, 1]
        # control de tamaño: train completo submuestreado a las filas y positivos de "antes de T" (mismos vehículos)
        idx_pos, idx_neg = np.where(train["todo"] & (y == 1))[0], np.where(train["todo"] & (y == 0))[0]
        ctrl = []
        for rep in range(reps):
            rng = np.random.default_rng(rep)
            r = np.concatenate([rng.choice(idx_pos, pos_pre, replace=False),
                                rng.choice(idx_neg, n_pre - pos_pre, replace=False)])
            ctrl.append(fit_gbm(f.loc[r, cols], y[r]).predict_proba(f[cols])[:, 1])
        preds["todo_submuestreado"] = np.mean(ctrl, 0)
        out = {"train_rows": {"todo": int(train["todo"].sum()), "antes_de_T": n_pre},
               "train_pos": {"todo": int(y[train["todo"]].sum()), "antes_de_T": pos_pre}}
        for per, sel in (("antes_de_T", ~post), ("desde_T", post), ("todo", np.ones_like(post))):
            te = tev & m & sel
            yt, v = y[te], f["v"].values[te]
            out[f"eval_{per}"] = {"n_rows": int(te.sum()), "n_vehicles": int(len(np.unique(v))),
                                  "n_event_vehicles": int(len(np.unique(v[yt == 1]))), "base_rate": float(yt.mean())}
            for name, p in preds.items():
                ci, _ = cluster_ci(yt, p[te], v, n=200)
                out[f"eval_{per}"][name] = {"auc": roc_auc_score(yt, p[te]), "auc_ci95": ci}
            out[f"eval_{per}"]["delta_antes_de_T_vs_todo"] = paired_ci(yt, preds["todo"][te], preds["antes_de_T"][te], v)
            out[f"eval_{per}"]["delta_submuestreado_vs_todo"] = paired_ci(yt, preds["todo"][te],
                                                                          preds["todo_submuestreado"][te], v)
        res[f"H{h}"] = out
        print(f"H{h}: train todo {out['train_rows']['todo']} filas / {out['train_pos']['todo']} pos; "
              f"antes de T {n_pre} / {pos_pre}", flush=True)
        for per in ("antes_de_T", "desde_T", "todo"):
            e = out[f"eval_{per}"]
            print(f"  evalúa {per:10s} ({e['n_vehicles']} veh, {e['n_event_vehicles']} con evento, base {e['base_rate']:.3f}): "
                  + " | ".join(f"{n} {e[n]['auc']:.3f} {np.round(e[n]['auc_ci95'], 3)}" for n in preds)
                  + f" | Δ antes_de_T {e['delta_antes_de_T_vs_todo'][0]:+.3f} {e['delta_antes_de_T_vs_todo'][1]}"
                  + f" | Δ submuestreado {e['delta_submuestreado_vs_todo'][0]:+.3f} {e['delta_submuestreado_vs_todo'][1]}",
                  flush=True)
    json.dump(res, open("data/temporal_decomp.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    learning() if "learning" in sys.argv else temporal_decomp()
