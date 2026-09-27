"""Evaluación en holdout de vehículos nunca vistos + explicabilidad global + perfiles de conductor.

Salidas: data/metrics.json, data/shap_global.parquet, data/profiles.parquet, data/leadtime.parquet
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import HORIZONS, feature_cols, outage

MODELS = ["gbm", "gru", "rsf", "stack"]
FPRS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
LOOKBACK = 180  # días antes del evento en los que una alarma cuenta como anticipación
COOLDOWN = 30  # días sin re-notificar tras una alerta
SCORE = "stack90"
OPERATING_FPR = 0.10  # punto de operación: misma tasa de falsas alarmas que la advertencia ECU actual

GROUPS = {  # hipótesis física -> prefijos de features
    "Hollín / DPF": ["acc_", "soot", "sh_full", "sh_over", "sh_overloaded", "sh_end_full", "sh_end_over", "life_sh_over"],
    "Regeneraciones": ["regen", "dbr", "km_per_regen", "days_since_regen", "km_since_regen", "sh_trip_end_in_regen"],
    "Patrón de uso": ["short", "micro", "km_per_trip", "n_trips", "km", "mins", "speed", "urban", "night", "idle", "active"],
    "Térmico / arranques en frío": ["etmax", "etavg", "cool", "never_warm", "cold_start"],
    "Consumo": ["fuel"],
    "Aceite": ["oil"],
    "Clima": ["air"],
    "Vehículo / mercado": ["Engine", "ModelSeries", "country"],
}


def group_of(c):
    body = c.split("_", 1)[1] if c.startswith(("w7_", "w30_", "w90_", "trend_", "trend30_", "selfz_", "life_")) else c
    body = body.replace("w30_", "").replace("w7_", "").replace("w90_", "")
    for g, keys in GROUPS.items():
        if any(k in body for k in keys) or any(c.startswith(k) for k in keys):
            return g
    return "Otros"


def smooth(p, score):
    """Media móvil de 7 días de la probabilidad por vehículo (reduce alarmas espurias)."""
    s = p.set_index("day").groupby("v")[score].rolling("7D").mean()
    return s.reset_index(level=0, drop=True).values


def leadtime(p, score, thr):
    """Por evento de vehículos fallados: ¿hubo alarma en los LOOKBACK días previos? ¿con cuánta anticipación?"""
    rows = []
    fa = p[p["failed"] == 1]
    for v, g in fa.groupby("v"):
        for ev in sorted(set((g["day"] + pd.to_timedelta(g["tte"], "D"))[g["tte"].notna()])):
            w = g[(g["day"] >= ev - pd.Timedelta(days=LOOKBACK)) & (g["day"] < ev)]
            if len(w) < 5:
                continue
            al = w[w[score] >= thr]
            rows.append({"v": v, "event": ev, "detected": len(al) > 0,
                         "lead_days": (ev - al["day"].min()).days if len(al) else 0})
    lt = pd.DataFrame(rows)
    h = p[p["failed"] == 0].copy()
    # episodio de alerta = día en alarma sin otra alerta emitida en los COOLDOWN días previos (no se re-notifica)
    on = h[h[score] >= thr]
    gap = on.groupby("v")["day"].diff().dt.days
    starts = int((gap.isna() | (gap > COOLDOWN)).sum())
    years = h.groupby("v")["day"].agg(lambda d: (d.max() - d.min()).days + 1).sum() / 365
    return lt, {"detection_rate": lt["detected"].mean(), "median_lead_days": lt.loc[lt.detected, "lead_days"].median(),
                "mean_lead_days": lt.loc[lt.detected, "lead_days"].mean(), "healthy_day_alarm_rate": (h[score] >= thr).mean(),
                "false_alarm_episodes_per_vehicle_year": starts / years, "n_events": len(lt)}


def main():
    p = pd.read_parquet("data/preds.parquet")
    f = pd.read_parquet("data/features.parquet")
    p["day"] = pd.to_datetime(p["day"])
    # evaluación solo antes del corte de telemetría de regeneraciones (igual que el entrenamiento)
    p = p[p["day"] < outage()].sort_values(["v", "day"]).reset_index(drop=True)
    f = f.set_index(["v", "day"]).loc[pd.MultiIndex.from_frame(p[["v", "day"]])].reset_index()
    test, oof = p["fold"] == -1, p["fold"] >= 0
    M = {"n_vehicles_test": int(p.loc[test, "v"].nunique()), "n_vehicles_train": int(p.loc[oof, "v"].nunique())}

    # ---- discriminación por modelo / horizonte ----
    disc = []
    for h in HORIZONS:
        m = test & p[f"m{h}"]
        y = p.loc[m, f"y{h}"]
        wf = m & (p["failed"] == 1)
        for mod in MODELS:
            s = p.loc[m, f"{mod}{h}"]
            disc.append({"model": mod, "H": h, "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                         "auc_within_failed": roc_auc_score(p.loc[wf, f"y{h}"], p.loc[wf, f"{mod}{h}"]),
                         "base_rate": y.mean(), "brier": brier_score_loss(y, s.clip(0, 1))})
        s = -np.log(p.loc[m, "ae_err"])
        disc.append({"model": "autoencoder", "H": h, "auc": roc_auc_score(y, -s), "ap": average_precision_score(y, -s),
                     "auc_within_failed": roc_auc_score(p.loc[wf, f"y{h}"], p.loc[wf, "ae_err"]), "base_rate": y.mean()})
    M["discrimination"] = disc
    # concordancia de la supervivencia (C-index) sobre el RUL
    from sksurv.metrics import concordance_index_censored
    m = test & p["usable"] & ((p["tte"].notna()) | (p["gap_to_end"] > 0))
    t = np.where(p.loc[m, "tte"].notna(), p.loc[m, "tte"], p.loc[m, "gap_to_end"])
    M["rsf_c_index"] = concordance_index_censored(p.loc[m, "tte"].notna().values, t, -p.loc[m, "rul"].values)[0]

    # ---- calibración del stack ----
    m = test & p["m90"]
    bins = pd.qcut(p.loc[m, SCORE], 10, duplicates="drop")
    M["calibration"] = p.loc[m].groupby(bins, observed=True).agg(pred=(SCORE, "mean"), obs=("y90", "mean")).to_dict("records")

    # ---- lead time vs falsas alarmas (umbral fijado en OOF sanos, aplicado a test) ----
    p["score_s"] = smooth(p, SCORE)
    # línea base "reactiva": advertencia actual de la ECU (DPF sobre límite en la última semana)
    p["ecu_warning"] = (f["w7_sh_over"].fillna(0) > 0).astype(float).values
    ref = p.loc[oof & (p["failed"] == 0), "score_s"]
    lead, curve = [], []
    for fpr in FPRS:
        thr = ref.quantile(1 - fpr)
        lt, s = leadtime(p[test], "score_s", thr)
        curve.append({"target_fpr": fpr, "threshold": thr, **s})
        lt["target_fpr"] = str(fpr)
        lead.append(lt)
    lt, s = leadtime(p[test], "ecu_warning", 0.5)
    M["ecu_baseline"] = s
    lt["target_fpr"] = "ECU"
    lead.append(lt)
    M["leadtime_curve"] = curve
    pd.concat(lead).to_parquet("data/leadtime.parquet")
    M["operating_fpr"] = OPERATING_FPR
    # IC 95% bootstrap sobre eventos (holdout chico -> reportar incertidumbre)
    rng = np.random.default_rng(0)
    for name, key in [("model", str(OPERATING_FPR)), ("ecu", "ECU")]:
        e = pd.concat(lead)
        e = e[e["target_fpr"] == key].reset_index(drop=True)
        bs = [e.iloc[rng.integers(0, len(e), len(e))] for _ in range(1000)]
        det = [b["detected"].mean() for b in bs]
        ld = [b.loc[b.detected, "lead_days"].median() for b in bs]
        M[f"ci95_{name}"] = {"detection": np.percentile(det, [2.5, 97.5]).tolist(),
                             "median_lead_days": np.nanpercentile(ld, [2.5, 97.5]).tolist()}
    M["alarm_threshold"] = next(c["threshold"] for c in curve if c["target_fpr"] == OPERATING_FPR)

    # ---- SHAP global (GBM 90d) ----
    booster = lgb.Booster(model_file="models/gbm90.txt")
    cols = feature_cols(f)
    sample = f.loc[test].sample(5000, random_state=0)[cols]
    sv = shap.TreeExplainer(booster).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    g = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0)})
    g["group"] = g["feature"].map(group_of)
    g.sort_values("mean_abs_shap", ascending=False).to_parquet("data/shap_global.parquet")
    M["shap_by_group"] = g.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).to_dict()

    # ---- perfiles de conductor (clustering de uso por vehículo, antes del primer evento) ----
    prof_cols = ["w90_sh_short5", "w90_km_per_trip", "w90_speed", "w90_sh_urban", "w90_km", "w90_sh_cold_start",
                 "w90_sh_never_warm", "w90_sh_idle", "w90_sh_night"]
    pre = f[(f["tte"].notna()) | (f["failed"] == 0)]
    veh = pre.groupby("v")[prof_cols].median().dropna()
    veh["failed"] = f.groupby("v")["failed"].first()
    Z = StandardScaler().fit_transform(veh[prof_cols])
    km = KMeans(4, n_init=20, random_state=0).fit(Z)
    veh["cluster"] = km.labels_
    cent = veh.groupby("cluster")[prof_cols + ["failed"]].mean()
    names, free = {}, set(cent.index)
    for crit, name in [("w90_sh_short5", "Urbano de viajes cortos"), ("w90_km_per_trip", "Ruta / larga distancia"),
                       ("w90_km", "Uso intensivo (alto km/día)")]:
        c = cent.loc[list(free), crit].idxmax()
        names[c] = name
        free.remove(c)
    names.update({c: "Mixto" for c in free})
    veh["profile"] = veh["cluster"].map(names)
    veh.reset_index().to_parquet("data/profiles.parquet")
    M["profiles"] = veh.groupby("profile").agg(n=("failed", "size"), failure_rate=("failed", "mean")).to_dict("index")

    p[["v", "day", "score_s", "ecu_warning"]].to_parquet("data/scores.parquet")
    json.dump(M, open("data/metrics.json", "w"), indent=2, default=float)
    print(pd.DataFrame(disc).round(3).to_string())
    print("C-index RSF:", round(M["rsf_c_index"], 3))
    print(pd.DataFrame(curve).round(3).to_string())
    print("CI95:", M["ci95_model"], M["ci95_ecu"])
    print("ECU baseline:", {k: round(v, 3) for k, v in M["ecu_baseline"].items()})
    print(pd.Series(M["shap_by_group"]).round(3))
    print(M["profiles"])


if __name__ == "__main__":
    main()
