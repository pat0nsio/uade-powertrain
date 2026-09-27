"""Validación temporal: simula poner el sistema en producción en la fecha T.

Entrenamiento: solo filas cuya etiqueta ya se conocía en T (día + H < T, o evento ocurrido antes de T).
Evaluación: filas con día >= T. Dos escenarios:
  * "vehículos nuevos": se entrena con vehículos de train y se evalúa en el holdout (vehículo + tiempo)
  * "misma flota":      se entrena y evalúa con todos los vehículos (la flota ya monitoreada, en el futuro)
El umbral de alerta se fija con un split temporal interno (sanos en [T-90, T)), sin mirar el período evaluado.
Se evalúa el LightGBM (componente principal del ensamble). Salida: data/temporal.json
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import FPRS, REL_PCTS, cluster_ci, fleet_threshold, leadtime, pick_operating, smooth
from src.features import HORIZONS, feature_cols
from src.models import GBM_PARAMS

T = pd.Timestamp("2026-01-01")
CAL = 90  # días previos a T usados para fijar el umbral


def known_at(f, h, t):
    """Filas cuya etiqueta y{h} era observable en la fecha t."""
    ev_day = f["day"] + pd.to_timedelta(f["tte"], "D")
    return f[f"m{h}"] & ((f["day"] + pd.Timedelta(days=h) < t) | ((f[f"y{h}"] == 1) & (ev_day < t)))


def fit(f, rows, h, cols):
    return lgb.LGBMClassifier(**GBM_PARAMS).fit(f.loc[rows, cols], f.loc[rows, f"y{h}"])


def scenario(f, fold, train_veh, test_veh, cols):
    out = {}
    for h in HORIZONS:
        m = fit(f, train_veh & known_at(f, h, T), h, cols)
        te = test_veh & (f["day"] >= T) & f[f"m{h}"]
        s = m.predict_proba(f.loc[te, cols])[:, 1]
        auc_ci, ap_ci = cluster_ci(f.loc[te, f"y{h}"], s, f.loc[te, "v"])
        out[f"H{h}"] = {"auc": roc_auc_score(f.loc[te, f"y{h}"], s), "auc_ci95": auc_ci,
                        "ap": average_precision_score(f.loc[te, f"y{h}"], s), "ap_ci95": ap_ci,
                        "base_rate": f.loc[te, f"y{h}"].mean(), "n_rows": int(te.sum()),
                        "n_vehicles": int(f.loc[te, "v"].nunique())}

    # alertas H=90 vs ECU en el período futuro. Los puntos de operación (umbral fijo y relativo) se eligen en el
    # período de calibración [T-90, T) con un modelo entrenado solo con lo conocido en T-90.
    t0 = T - pd.Timedelta(days=CAL)
    inner = fit(f, train_veh & known_at(f, 90, t0), 90, cols)
    cal = train_veh & (f["day"] >= t0) & (f["day"] < T)
    c = f.loc[cal, ["v", "day", "tte"]].copy()
    ev_day = c["day"] + pd.to_timedelta(c["tte"], "D")
    c.loc[ev_day >= T, "tte"] = np.nan  # en T todavía no se conocen eventos futuros
    known_failed = set(c.loc[c["tte"].notna(), "v"])
    c["failed"] = c["v"].isin(known_failed).astype(int)
    c["p"] = inner.predict_proba(f.loc[cal, cols])[:, 1]
    c["s"] = smooth(c, "p")
    c["ecu"] = (f.loc[cal, "w7_sh_over"].fillna(0) > 0).astype(float)
    ecu_cal = leadtime(c, "ecu", 0.5)[1]
    fix_thr = {fpr: c.loc[c["failed"] == 0, "s"].quantile(1 - fpr) for fpr in FPRS}
    fix_op = pick_operating([{"target_fpr": x, **leadtime(c, "s", t)[1]} for x, t in fix_thr.items()], ecu_cal)
    rel_op = pick_operating([{"pct": x, **leadtime(c.assign(a=(c["s"] >= fleet_threshold(c, "s", x)).astype(float)),
                                                   "a", 0.5)[1]} for x in REL_PCTS], ecu_cal)

    final = fit(f, train_veh & known_at(f, 90, T), 90, cols)
    fut = test_veh & (f["day"] >= T)
    q = f.loc[fut, ["v", "day", "failed", "tte"]].copy()
    q["p"] = final.predict_proba(f.loc[fut, cols])[:, 1]
    q["s"] = smooth(q, "p")
    q["ecu"] = (f.loc[fut, "w7_sh_over"].fillna(0) > 0).astype(float)
    ecu = leadtime(q, "ecu", 0.5)[1]
    fixed = [{"target_fpr": x, **leadtime(q, "s", t)[1]} for x, t in fix_thr.items()]
    relative = [{"pct": x, **leadtime(q.assign(a=(q["s"] >= fleet_threshold(q, "s", x)).astype(float)), "a", 0.5)[1]}
                for x in REL_PCTS]
    out["alerting_H90"] = {"ecu": ecu, "fixed_curve": fixed, "relative_curve": relative,
                           "fixed_operating": fix_op, "relative_operating": rel_op,
                           "fixed_at_op": next(x for x in fixed if x["target_fpr"] == fix_op),
                           "relative_at_op": next(x for x in relative if x["pct"] == rel_op)}
    return out


def main():
    f = pd.read_parquet("data/features.parquet")
    f["day"] = pd.to_datetime(f["day"])
    f = f.sort_values(["v", "day"]).reset_index(drop=True)
    p = pd.read_parquet("data/preds.parquet", columns=["v", "fold"]).drop_duplicates("v").set_index("v")["fold"]
    fold = f["v"].map(p)
    cols = feature_cols(f)
    every = pd.Series(True, index=f.index)
    R = {"T": str(T.date()),
         "vehiculos_nuevos": scenario(f, fold, fold >= 0, fold == -1, cols),
         "misma_flota": scenario(f, fold, every, every, cols)}
    json.dump(R, open("data/temporal.json", "w"), indent=2, default=float)
    for k in ("vehiculos_nuevos", "misma_flota"):
        print(f"== {k} (entrena < {R['T']}, evalúa >=)")
        for h in HORIZONS:
            r = R[k][f"H{h}"]
            print(f"  H{h}: AUC {r['auc']:.3f} {np.round(r['auc_ci95'], 3)}  AP {r['ap']:.3f} (base {r['base_rate']:.3f})"
                  f"  filas={r['n_rows']} vehículos={r['n_vehicles']}")
        a = R[k]["alerting_H90"]
        r = lambda d: {x: round(y, 3) for x, y in d.items()}
        print("  ECU:            ", r(a["ecu"]))
        print("  umbral fijo op: ", r(a["fixed_at_op"]))
        print("  umbral rel. op: ", r(a["relative_at_op"]))


if __name__ == "__main__":
    main()
