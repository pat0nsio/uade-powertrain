"""Features de ventana móvil (7/30/90 días, solo pasado) + etiquetas de tiempo-al-evento.

Salidas:
  data/calendar.parquet  vehículo x día calendario (incluye días sin uso) -> entrada de la red secuencial
  data/features.parquet  una fila por vehículo-día activo (punto de predicción) con features + labels
"""
import json

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90)
POST_EVENT_BLACKOUT = 14  # días post-service excluidos (transición / reparación)

SUMS = ["n_trips", "km", "mins", "fuel_used", "soot_delta", "oil_drop", "regen_completed_in_trip",
        "n_regen", "n_regen_stopped", "n_manual_regen", "n_msgs"]
TRIP_W = ["sh_short5", "sh_short10", "sh_micro", "sh_urban", "sh_never_warm", "sh_cold_start",
          "sh_trip_end_in_regen", "sh_end_full", "sh_end_over", "sh_night", "sh_idle",
          "etmax", "etavg", "cool_rise", "air", "soot_mean"]
MSG_W = ["acc_mean", "sh_full", "sh_over", "sh_overloaded", "sh_regen_msg"]
MAXS = ["soot_max", "acc_max"]
MINS = ["dbr_min", "oil_min", "air_min"]
CATS = ["Engine", "ModelSeries", "country"]
WINDOWS = (7, 30, 90)
# Nunca como feature: identifican el vehículo/cohorte o el tiempo absoluto (sesgo de muestreo: fallados son más viejos)
NON_FEATURES = {"v", "day", "failed", "tte", "gap_to_end", "age_days", "usable", *[f"y{h}" for h in HORIZONS],
                *[f"m{h}" for h in HORIZONS]}


REGEN_DERIVED = ["n_regen", "km_per_regen", "regen_stop_ratio"]


def outage():
    return pd.Timestamp(json.load(open("data/quality.json"))["regen_telemetry_outage_from"])


def calendar(daily, static):
    """Reindexa cada vehículo a días calendario consecutivos; días sin uso quedan en 0/NaN."""
    out = []
    for v, g in daily.groupby("v", sort=False):
        idx = pd.date_range(g["day"].min(), g["day"].max(), freq="D")
        g = g.set_index("day").reindex(idx)
        g["v"] = v
        g["active"] = g["n_trips"].notna() | g["n_msgs"].notna()
        g[SUMS] = g[SUMS].fillna(0)
        out.append(g.rename_axis("day").reset_index())
    cal = pd.concat(out, ignore_index=True)
    cal.loc[cal["day"] >= outage(), "n_regen"] = np.nan  # corte de telemetría: faltante, no cero
    return cal


def rolling(cal):
    g = cal.groupby("v", sort=False)
    f = pd.DataFrame(index=cal.index)
    wt = cal[TRIP_W].mul(cal["n_trips"], axis=0).fillna(0)
    wm = cal[MSG_W].mul(cal["n_msgs"], axis=0).fillna(0)
    wt.columns, wm.columns = [c + "__w" for c in TRIP_W], [c + "__w" for c in MSG_W]
    tmp = pd.concat([cal[["v"] + SUMS], wt, wm, cal[MAXS + MINS], cal["active"].astype(float)], axis=1)
    gt = tmp.groupby("v", sort=False)
    for w in WINDOWS:
        r = gt.rolling(w, min_periods=1)
        s = r[SUMS + list(wt.columns) + list(wm.columns) + ["active"]].sum().reset_index(level=0, drop=True)
        mx = r[MAXS].max().reset_index(level=0, drop=True)
        mn = r[MINS].min().reset_index(level=0, drop=True)
        p = f"w{w}_"
        for c in SUMS:
            f[p + c] = s[c] / w  # por día
        for c in TRIP_W:
            f[p + c] = s[c + "__w"] / s["n_trips"].replace(0, np.nan)
        for c in MSG_W:
            f[p + c] = s[c + "__w"] / s["n_msgs"].replace(0, np.nan)
        for c in MAXS:
            f[p + c] = mx[c]
        for c in MINS:
            f[p + c] = mn[c]
        f[p + "active_share"] = s["active"] / w
        f[p + "speed"] = s["km"] / (s["mins"] / 60).replace(0, np.nan)
        f[p + "fuel_per100"] = s["fuel_used"] / s["km"].replace(0, np.nan) * 100
        f[p + "km_per_regen"] = s["km"] / s["n_regen"].replace(0, np.nan)
        f[p + "km_per_trip"] = s["km"] / s["n_trips"].replace(0, np.nan)
        f[p + "regen_stop_ratio"] = s["n_regen_stopped"] / (s["n_regen"] + 1)
        f[p + "soot_per_km"] = s["soot_delta"].clip(lower=0) / s["km"].replace(0, np.nan)

        # ventana que toca el corte de telemetría de regeneraciones -> faltante
        hit = cal["day"] > outage() - pd.Timedelta(days=w)
        f.loc[hit, [p + c for c in REGEN_DERIVED]] = np.nan

    # tendencias: corto plazo vs largo plazo (aceleración de la degradación)
    for c in ["acc_mean", "sh_over", "sh_full", "km_per_regen", "fuel_per100", "sh_short5", "soot_mean", "n_regen", "speed"]:
        f[f"trend_{c}"] = f[f"w7_{c}"] - f[f"w90_{c}"]
        f[f"trend30_{c}"] = f[f"w30_{c}"] - f[f"w90_{c}"]

    # estado: km / días desde la última regeneración (se reinicia con cada regen)
    regen = cal["n_regen"] > 0
    grp = regen.groupby(cal["v"]).cumsum()
    f["km_since_regen"] = cal["km"].groupby([cal["v"], grp]).cumsum()
    f["days_since_regen"] = cal.groupby([cal["v"], grp]).cumcount()
    f.loc[(grp == 0) | (cal["day"] >= outage()), ["km_since_regen", "days_since_regen"]] = np.nan

    # desvío respecto de la propia historia del vehículo (solo pasado: expanding desplazado)
    for c in ["w30_acc_mean", "w30_sh_over", "w30_km_per_regen", "w30_fuel_per100", "w30_sh_short5"]:
        hist = f[c].groupby(cal["v"]).transform(lambda s: s.expanding(30).mean().shift(30))
        sd = f[c].groupby(cal["v"]).transform(lambda s: s.expanding(30).std().shift(30))
        f[f"selfz_{c}"] = (f[c] - hist) / sd.replace(0, np.nan)

    # tasas de mal uso a lo largo de la vida (normalizadas -> no crecen con la edad del vehículo)
    cum = lambda x: x.groupby(cal["v"]).cumsum()
    f["life_sh_short5"] = cum(cal["sh_short5"].fillna(0) * cal["n_trips"]) / cum(cal["n_trips"]).replace(0, np.nan)
    f["life_sh_over"] = cum(cal["sh_over"].fillna(0) * cal["n_msgs"]) / cum(cal["n_msgs"]).replace(0, np.nan)
    f["life_regen_stopped_per_1000km"] = cum(cal["n_regen_stopped"]) / cum(cal["km"]).replace(0, np.nan) * 1000
    f["life_km_per_regen"] = cum(cal["km"]) / cum(cal["n_regen"].fillna(0)).replace(0, np.nan)
    f.loc[cal["day"] >= outage(), "life_km_per_regen"] = np.nan
    return f


def labels(cal, static):
    ev = static.set_index("v")["events"]
    last = cal.groupby("v")["day"].transform("max")
    tte = np.full(len(cal), np.nan)
    since = np.full(len(cal), np.inf)
    days = cal["day"].values.astype("datetime64[D]")
    for v, idx in cal.groupby("v").indices.items():
        e = np.sort(np.array(list(ev.get(v, [])), dtype="datetime64[D]"))
        if not len(e):
            continue
        d = days[idx]
        k = np.searchsorted(e, d)  # próximo evento >= d
        nxt = np.where(k < len(e), e[np.minimum(k, len(e) - 1)], np.datetime64("NaT"))
        tte[idx] = np.where(k < len(e), (nxt - d).astype("timedelta64[D]").astype(float), np.nan)
        prev = np.where(k > 0, e[np.maximum(k - 1, 0)], np.datetime64("NaT"))
        since[idx] = np.where(k > 0, (d - prev).astype("timedelta64[D]").astype(float), np.inf)
    y = pd.DataFrame({"tte": tte, "gap_to_end": (last - cal["day"]).dt.days.values}, index=cal.index)
    # corte administrativo simétrico: solo días previos al corte de telemetría (ambas cohortes por igual)
    usable = (y["tte"] != 0) & (since > POST_EVENT_BLACKOUT) & (cal["day"] < outage()).values
    y["usable"] = usable
    for h in HORIZONS:
        y[f"y{h}"] = (y["tte"] <= h).astype(int)
        # negativo solo si el futuro de h días es observable (censura)
        y[f"m{h}"] = usable & ((y[f"y{h}"] == 1) | (y["gap_to_end"] >= h))
    return y


def build():
    daily = pd.read_parquet("data/daily.parquet")
    static = pd.read_parquet("data/static.parquet")
    daily["day"] = pd.to_datetime(daily["day"])
    cal = calendar(daily, static)
    f = rolling(cal)
    y = labels(cal, static)
    st = static.set_index("v")
    base = cal[["v", "day", "active"]].copy()
    base["failed"] = base["v"].map(st["failed"])
    base["age_days"] = (base["day"] - pd.to_datetime(base["v"].map(st["prod"]))).dt.days
    for c in CATS:
        base[c] = base["v"].map(st[c]).astype("category")
    feats = pd.concat([base, f, y], axis=1)
    feats = feats[feats["active"]].drop(columns="active").reset_index(drop=True)
    cal.to_parquet("data/calendar.parquet")
    feats.to_parquet("data/features.parquet")
    return feats


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


if __name__ == "__main__":
    feats = build()
    cols = feature_cols(feats)
    print(f"{len(feats)} filas, {len(cols)} features")
    for h in HORIZONS:
        m = feats[f"m{h}"]
        print(f"H={h}: usable={m.sum()} pos={feats.loc[m, f'y{h}'].sum()} rate={feats.loc[m, f'y{h}'].mean():.4f}")
    # check anti-leakage: ninguna feature es función del futuro -> recomputar sobre historia truncada debe dar igual
    daily = pd.read_parquet("data/daily.parquet"); daily["day"] = pd.to_datetime(daily["day"])
    static = pd.read_parquet("data/static.parquet")
    v = feats.loc[feats["failed"] == 1, "v"].iloc[0]
    dv = daily[daily["v"] == v]
    cut = dv["day"].iloc[len(dv) // 2]
    full = rolling(calendar(dv, static))
    trunc = rolling(calendar(dv[dv["day"] <= cut], static))
    num = full.select_dtypes("number").columns
    pd.testing.assert_frame_equal(full.iloc[:len(trunc)][num], trunc[num], check_dtype=False)
    print("anti-leakage OK (features idénticas con historia truncada)")
