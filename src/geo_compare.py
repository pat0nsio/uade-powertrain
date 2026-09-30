"""Experimento rápido (solo LightGBM, sin red neuronal): país vs altitud de la ciudad de venta.

Requiere features construidas con ambas variables:
    DPF_GEO=both python -m src.features
    python -m src.geo_compare

Compara tres conjuntos de features con el mismo split (holdout = fold -1, OOF = folds 0..4) y el mismo protocolo que
src/evaluate.py, pero con el GBM a 90 días como score de alerta (no el stack):
  pais     = features actuales (con country, sin altitud)
  altitud  = altitud monótona creciente en lugar de country
  ambos    = country + altitud monótona
Reporta AUC holdout (IC95 por vehículo), AUC OOF, AUC dentro de fallados, Δ AUC de a pares contra "pais", y la alerta
(umbral elegido en OOF con falsas alarmas <= ECU) en el holdout, más un chequeo de cohorte: qué fracción de los días
de fallados LEJOS del evento (más de 180 días antes) queda en alerta. Si se acerca a la detección, el modelo reconoce "autos que
fallan" en vez de detectar degradación.
Salida: data/geo_compare.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import FPRS, LOOKBACK, cluster_ci, leadtime, smooth
from src.features import ALT, HORIZONS, feature_cols
from src.gbm import K, fit_gbm, split


def paired_ci(y, a, b, v, n=200, seed=0):
    """Δ AUC (b − a) e IC95 remuestreando vehículos (mismas filas para los dos modelos). Igual que src.diagnose."""
    rng = np.random.default_rng(seed)
    g = list(pd.Series(np.arange(len(y))).groupby(v).indices.values())
    d = []
    for _ in range(n):
        i = np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))])
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).round(4).tolist()


def oof_and_holdout(f, cols, h):
    """Predicciones OOF para los folds de train y promedio de los K modelos para el holdout."""
    pred = np.full(len(f), np.nan)
    m = f[f"m{h}"].values
    test = f["fold"].values == -1
    for k in range(K):
        tr = (f["fold"].values >= 0) & (f["fold"].values != k) & m
        va = f["fold"].values == k
        mod = fit_gbm(f.loc[tr, cols], f.loc[tr, f"y{h}"])
        pred[va] = mod.predict_proba(f.loc[va, cols])[:, 1]
        hold = mod.predict_proba(f.loc[test, cols])[:, 1] / K
        pred[test] = hold if k == 0 else pred[test] + hold
    return pred


def alert_report(p, oof, test):
    """Punto de operación elegido en OOF (falsas alarmas <= ECU) y aplicado al holdout."""
    ecu_oof = leadtime(p[oof], "ecu_warning", 0.5)[1]
    ref = p.loc[oof & (p["failed"] == 0), "score_s"]
    ok = [fpr for fpr in FPRS if leadtime(p[oof], "score_s", ref.quantile(1 - fpr))[1]
          ["false_alarm_episodes_per_vehicle_year"] <= ecu_oof["false_alarm_episodes_per_vehicle_year"]]
    op = max(ok) if ok else min(FPRS)
    thr = ref.quantile(1 - op)
    s = leadtime(p[test], "score_s", thr)[1]
    far = test & (p["failed"] == 1) & (p["tte"] > LOOKBACK)  # antes del evento, fuera de la ventana de anticipación
    return {"operating_fpr": op, "detection": s["detection_rate"], "median_lead_days": s["median_lead_days"],
            "false_alarms_per_vehicle_year": s["false_alarm_episodes_per_vehicle_year"],
            "healthy_day_alarm_rate": s["healthy_day_alarm_rate"],
            "far_failed_day_alarm_rate": float((p.loc[far, "score_s"] >= thr).mean())}


def main():
    f = pd.read_parquet("data/features.parquet")
    assert "country" in f and ALT in f, "construir antes las features con: DPF_GEO=both python -m src.features"
    f["day"] = pd.to_datetime(f["day"])
    f = f.sort_values(["v", "day"]).reset_index(drop=True)
    f["fold"] = split(f)
    base = [c for c in feature_cols(f) if c not in ("country", ALT)]
    variants = {"pais": base + ["country"], "altitud": base + [ALT], "ambos": base + ["country", ALT]}
    test, oof = f["fold"] == -1, f["fold"] >= 0
    ecu = (f["w7_sh_over"].fillna(0) > 0).astype(float)
    res, preds = {"n_vehicles_test": int(f.loc[test, "v"].nunique()),
                  "altitude_coverage": float(f.groupby("v")[ALT].first().notna().mean())}, {}
    for name, cols in variants.items():
        print(f"== {name}: {len(cols)} features", flush=True)
        preds[name] = {h: oof_and_holdout(f, cols, h) for h in HORIZONS}
        r = {}
        for h in HORIZONS:
            m, pr = f[f"m{h}"], preds[name][h]
            t, o, wf = test & m, oof & m, test & m & (f["failed"] == 1)
            r[f"H{h}"] = {"auc": roc_auc_score(f.loc[t, f"y{h}"], pr[t]),
                          "auc_ci95": cluster_ci(f.loc[t, f"y{h}"], pr[t], f.loc[t, "v"], n=200)[0],
                          "oof_auc": roc_auc_score(f.loc[o, f"y{h}"], pr[o]),
                          "auc_within_failed": roc_auc_score(f.loc[wf, f"y{h}"], pr[wf])}
            if name != "pais":
                y, a, b = f.loc[t, f"y{h}"].values, preds["pais"][h][t], pr[t]
                r[f"H{h}"]["delta_auc_vs_pais"] = paired_ci(y, a, b, f.loc[t, "v"].values, n=200)
        p = f[["v", "day", "failed", "tte"]].assign(gbm90=preds[name][90], ecu_warning=ecu.values)
        p["score_s"] = smooth(p, "gbm90")
        r["alert"] = alert_report(p, oof, test)
        if name == "pais":
            s = leadtime(p[test], "ecu_warning", 0.5)[1]
            res["ecu"] = {"detection": s["detection_rate"], "median_lead_days": s["median_lead_days"],
                          "false_alarms_per_vehicle_year": s["false_alarm_episodes_per_vehicle_year"]}
        res[name] = r
        print(" ".join(f"H{h} AUC {r[f'H{h}']['auc']:.3f} (OOF {r[f'H{h}']['oof_auc']:.3f})" for h in HORIZONS),
              "| detección {detection:.0%}, FA/veh-año {false_alarms_per_vehicle_year:.2f}, "
              "días lejanos de fallados en alerta {far_failed_day_alarm_rate:.1%}".format(**r["alert"]), flush=True)
    print("ECU: detección {detection:.0%}, FA/veh-año {false_alarms_per_vehicle_year:.2f}".format(**res["ecu"]))
    json.dump(res, open("data/geo_compare.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
