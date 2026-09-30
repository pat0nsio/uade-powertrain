"""Evaluación en holdout de vehículos nunca vistos + explicabilidad global + perfiles de conductor.

Salidas: data/metrics.json, data/shap_global.parquet, data/profiles.parquet, data/leadtime.parquet
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import HORIZONS, feature_cols

MODELS = ["gbm", "gru", "rsf", "stack"]
FPRS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
LOOKBACK = 180  # días antes del evento en los que una alarma cuenta como anticipación
COOLDOWN = 30  # días sin re-notificar tras una alerta
SCORE = "stack90"

GROUPS = {  # hipótesis física -> prefijos de features
    "Hollín / DPF": ["acc_", "soot", "sh_full", "sh_over", "sh_overloaded", "sh_end_full", "sh_end_over", "life_sh_over"],
    "Regeneraciones": ["regen", "dbr", "km_per_regen", "days_since_regen", "km_since_regen", "sh_trip_end_in_regen"],
    "Patrón de uso": ["short", "micro", "km_per_trip", "n_trips", "km", "mins", "speed", "urban", "night", "idle", "active"],
    "Térmico / arranques en frío": ["etmax", "etavg", "cool", "never_warm", "cold_start"],
    "Consumo": ["fuel"],
    "Aceite": ["oil"],
    "Clima": ["air"],
    "Vehículo / mercado": ["Engine", "ModelSeries", "country", "altitud"],
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
            r = {"v": v, "event": ev, "detected": len(al) > 0, "lead_days": (ev - al["day"].min()).days if len(al) else 0}
            if "cum_km" in g:  # km recorridos entre la primera alerta y el evento
                r["lead_km"] = g.loc[g["day"] <= ev, "cum_km"].iloc[-1] - al["cum_km"].iloc[0] if len(al) else 0.0
            rows.append(r)
    lt = pd.DataFrame(rows)
    h = p[p["failed"] == 0].copy()
    # episodio de alerta = día en alarma sin otra alerta emitida en los COOLDOWN días previos (no se re-notifica)
    on = h[h[score] >= thr]
    gap = on.groupby("v")["day"].diff().dt.days
    starts = int((gap.isna() | (gap > COOLDOWN)).sum())
    years = h.groupby("v")["day"].agg(lambda d: (d.max() - d.min()).days + 1).sum() / 365
    out = {"detection_rate": lt["detected"].mean(), "median_lead_days": lt.loc[lt.detected, "lead_days"].median(),
           "mean_lead_days": lt.loc[lt.detected, "lead_days"].mean(), "healthy_day_alarm_rate": (h[score] >= thr).mean(),
           "false_alarm_episodes_per_vehicle_year": starts / years, "n_events": len(lt)}
    if "lead_km" in lt:
        out["median_lead_km"] = lt.loc[lt.detected, "lead_km"].median()
    return lt, out


REL_PCTS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
REL_WINDOW = 30  # días de historia de la flota para el umbral relativo


def fleet_threshold(p, score, pct, window=REL_WINDOW):
    """Umbral relativo causal: cuantil (1 - pct) del riesgo de TODA la flota en los últimos `window` días (hasta hoy).
    No usa etiquetas -> se puede calcular en producción, y se ajusta solo si la flota entera se desplaza."""
    d = p["day"].values.astype("datetime64[D]")
    order = np.argsort(d, kind="stable")
    ds, ss = d[order], p[score].values[order]
    days = np.unique(ds)
    lo = np.searchsorted(ds, days - np.timedelta64(window - 1, "D"))
    hi = np.searchsorted(ds, days, side="right")
    thr = np.array([np.nanquantile(ss[a if b - a >= 200 else 0:b], 1 - pct) for a, b in zip(lo, hi)])
    return thr[np.searchsorted(days, d)]


def pick_operating(curve, ecu):
    """Mayor sensibilidad cuya tasa de falsas alarmas no supera la de la ECU."""
    key = "target_fpr" if "target_fpr" in curve[0] else "pct"
    ok = [c[key] for c in curve if c["false_alarm_episodes_per_vehicle_year"] <= ecu["false_alarm_episodes_per_vehicle_year"]]
    return max(ok) if ok else min(c[key] for c in curve)


def cluster_ci(y, s, veh, n=300, seed=0):
    """IC 95% de AUC y AP remuestreando VEHÍCULOS (las filas de un mismo vehículo están correlacionadas)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    groups = list(pd.Series(np.arange(len(y))).groupby(np.asarray(veh)).indices.values())
    aucs, aps = [], []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], s[idx])); aps.append(average_precision_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5]).tolist(), np.percentile(aps, [2.5, 97.5]).tolist()


def main():
    p = pd.read_parquet("data/preds.parquet")
    f = pd.read_parquet("data/features.parquet")
    p["day"] = pd.to_datetime(p["day"])
    p = p.sort_values(["v", "day"]).reset_index(drop=True)
    f = f.set_index(["v", "day"]).loc[pd.MultiIndex.from_frame(p[["v", "day"]])].reset_index()
    test, oof = p["fold"] == -1, p["fold"] >= 0
    # km acumulados por vehículo (suma de los km diarios de los viajes) -> anticipación en km
    cal = pd.read_parquet("data/calendar.parquet", columns=["v", "day", "km"])
    cal["cum_km"] = cal.groupby("v")["km"].cumsum()
    p["cum_km"] = cal.set_index(["v", "day"])["cum_km"].reindex(pd.MultiIndex.from_frame(p[["v", "day"]])).values
    M = {"n_vehicles_test": int(p.loc[test, "v"].nunique()), "n_vehicles_train": int(p.loc[oof, "v"].nunique())}

    # ---- discriminación por modelo / horizonte ----
    disc = []
    for h in HORIZONS:
        m = test & p[f"m{h}"]
        y = p.loc[m, f"y{h}"]
        wf = m & (p["failed"] == 1)
        for mod in MODELS:
            s = p.loc[m, f"{mod}{h}"]
            auc_ci, ap_ci = cluster_ci(y, s, p.loc[m, "v"])
            disc.append({"model": mod, "H": h, "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                         "auc_ci95": auc_ci, "ap_ci95": ap_ci,
                         "oof_auc": roc_auc_score(p.loc[oof & p[f"m{h}"], f"y{h}"], p.loc[oof & p[f"m{h}"], f"{mod}{h}"]),
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
    # punto de operación elegido SOLO con OOF: el mayor FPR cuya tasa de falsas alarmas no supere la de la ECU
    ecu_oof = leadtime(p[oof], "ecu_warning", 0.5)[1]
    oof_curve = [{"target_fpr": fpr, **leadtime(p[oof], "score_s", ref.quantile(1 - fpr))[1]} for fpr in FPRS]
    ok = [c["target_fpr"] for c in oof_curve
          if c["false_alarm_episodes_per_vehicle_year"] <= ecu_oof["false_alarm_episodes_per_vehicle_year"]]
    op = max(ok) if ok else min(FPRS)
    M["oof_leadtime_curve"], M["oof_ecu_baseline"] = oof_curve, ecu_oof
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
    M["operating_fpr"] = op
    # IC 95% bootstrap sobre eventos (holdout chico -> reportar incertidumbre)
    rng = np.random.default_rng(0)
    for name, key in [("model", str(op)), ("ecu", "ECU")]:
        e = pd.concat(lead)
        e = e[e["target_fpr"] == key].reset_index(drop=True)
        bs = [e.iloc[rng.integers(0, len(e), len(e))] for _ in range(1000)]
        det = [b["detected"].mean() for b in bs]
        ld = [b.loc[b.detected, "lead_days"].median() for b in bs]
        lk = [b.loc[b.detected, "lead_km"].median() for b in bs]
        M[f"ci95_{name}"] = {"detection": np.percentile(det, [2.5, 97.5]).tolist(),
                             "median_lead_days": np.nanpercentile(ld, [2.5, 97.5]).tolist(),
                             "median_lead_km": np.nanpercentile(lk, [2.5, 97.5]).tolist()}
    M["alarm_threshold"] = next(c["threshold"] for c in curve if c["target_fpr"] == op)

    # ---- umbral RELATIVO a la flota (alerta = top X% de riesgo de la flota en los últimos 30 días) ----
    rel = {pct: (p["score_s"].values >= fleet_threshold(p, "score_s", pct)).astype(float) for pct in REL_PCTS}
    oof_rel = [{"pct": pct, **leadtime(p[oof].assign(a=rel[pct][oof.values]), "a", 0.5)[1]} for pct in REL_PCTS]
    rop = pick_operating(oof_rel, ecu_oof)
    rel_curve, rel_lead = [], None
    for pct in REL_PCTS:
        lt, s = leadtime(p[test].assign(a=rel[pct][test.values]), "a", 0.5)
        rel_curve.append({"pct": pct, **s})
        rel_lead = lt if pct == rop else rel_lead
    bs = [rel_lead.iloc[rng.integers(0, len(rel_lead), len(rel_lead))] for _ in range(1000)]
    M.update(oof_relative_curve=oof_rel, relative_curve=rel_curve, relative_operating_pct=rop,
             ci95_relative={"detection": np.percentile([b["detected"].mean() for b in bs], [2.5, 97.5]).tolist(),
                            **{f"median_{k}": np.nanpercentile([b.loc[b.detected, k].median() for b in bs],
                                                               [2.5, 97.5]).tolist() for k in ("lead_days", "lead_km")}})
    p["thr_rel"] = fleet_threshold(p, "score_s", rop)
    p["thr_rel_med"] = fleet_threshold(p, "score_s", min(2 * rop, 0.5))

    # ---- SHAP global (GBM 90d) ----
    booster = lgb.Booster(model_file="models/gbm90.txt")
    cols = feature_cols(f)
    sample = f.loc[test].sample(5000, random_state=0)[cols]
    import shap  # import diferido: el resto del módulo (métricas, alertas) no lo necesita
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

    p[["v", "day", "score_s", "ecu_warning", "thr_rel", "thr_rel_med"]].to_parquet("data/scores.parquet")
    json.dump(M, open("data/metrics.json", "w"), indent=2, default=float)
    print(pd.DataFrame(disc).round(3).to_string())
    print("C-index RSF:", round(M["rsf_c_index"], 3))
    print(pd.DataFrame(curve).round(3).to_string())
    print("punto de operación (elegido en OOF):", op, "| CI95:", M["ci95_model"], M["ci95_ecu"])
    print("umbral relativo (elegido en OOF): top", rop, "| CI95 detección:", M["ci95_relative"])
    print(pd.DataFrame(rel_curve).round(3).to_string())
    print("ECU baseline:", {k: round(v, 3) for k, v in M["ecu_baseline"].items()})
    print(pd.Series(M["shap_by_group"]).round(3))
    print(M["profiles"])


if __name__ == "__main__":
    main()
