"""Copiloto: receta mínima de cambio de hábito y ventana habitual para regenerar."""
import itertools

import numpy as np
import pandas as pd

MIN_MINS = 20  # trayecto apto: una regeneración en curso termina en > 90 % de los viajes de 20 min o más (completion)
WEEKS = 12  # historia usada para el patrón semanal
DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
BLOCKS = {"madrugada": (0, 6), "mañana": (6, 12), "tarde": (12, 18), "noche": (18, 24)}
LONG_KM, LONG_KMH = 40, 80  # supuesto del simulador: un viaje de ruta extra = 40 km a 80 km/h (30 min)
# esfuerzo relativo de cada cambio, para elegir la receta más fácil que alcanza
COST = {"warm": 1, "short_cut_25": 1, "long_trip": 2}


def simulate(X, long_trips=0, short_cut=0, warm=False):
    """Features del día con el hábito cambiado (solo las que tienen restricción monótona)."""
    Xs = X.copy()
    for wd in (7, 30, 90):
        n = Xs[f"w{wd}_n_trips"] * wd
        add = long_trips * wd / 7
        removed = Xs[f"w{wd}_sh_short5"] * n * short_cut / 100
        n2 = (n - removed + add).replace(0, np.nan)
        km = Xs[f"w{wd}_km"] * wd
        for c in ["sh_short5", "sh_short10", "sh_urban", "sh_never_warm", "sh_cold_start"]:
            Xs[f"w{wd}_{c}"] = ((Xs[f"w{wd}_{c}"] * n - removed).clip(lower=0) / n2).fillna(0)
        Xs[f"w{wd}_sh_micro"] = (Xs[f"w{wd}_sh_micro"] * n * (1 - short_cut / 100) / n2).fillna(0)
        Xs[f"w{wd}_km_per_trip"] = (km + LONG_KM * add) / n2
        hours = km / Xs[f"w{wd}_speed"].replace(0, np.nan)
        Xs[f"w{wd}_speed"] = ((km + LONG_KM * add) / (hours + add * LONG_KM / LONG_KMH)).fillna(Xs[f"w{wd}_speed"])
        if warm:
            Xs[f"w{wd}_sh_trip_end_in_regen"] = 0
            Xs[f"w{wd}_regen_stop_ratio"] = 0
    if long_trips:  # un trayecto largo completa una regeneración
        Xs[["days_since_regen", "km_since_regen"]] = 0
    return Xs


def min_recipe(model, X, target):
    """Combinación de menor esfuerzo con riesgo <= target; si ninguna alcanza, la de mayor reducción."""
    p0 = float(model.predict(X)[0])
    opts = []
    for lt, sc, w in itertools.product(range(4), (0, 25, 50, 75, 100), (False, True)):
        if lt == sc == 0 and not w:
            continue
        p = float(model.predict(simulate(X, lt, sc, w))[0])
        cost = lt * COST["long_trip"] + sc // 25 * COST["short_cut_25"] + w * COST["warm"]
        opts.append({"long_trips": lt, "short_cut": sc, "warm": w, "risk": p, "cost": cost})
    ok = [o for o in opts if o["risk"] <= target and o["risk"] < p0]
    best = min(ok, key=lambda o: (o["cost"], o["risk"])) if ok else min(opts, key=lambda o: (o["risk"], o["cost"]))
    return {**best, "risk0": p0, "target": target, "reaches": bool(ok), "already_ok": p0 <= target}


def recipe_text(r):
    if r["already_ok"]:
        return ("Según el modelo de hábitos, el riesgo ya está por debajo del umbral de la flota: cambiar el manejo "
                "no lo baja de forma apreciable. Si el vehículo está en alerta, revisar el filtro en el concesionario.")
    if not r["reaches"]:
        return (f"Ni combinando todos los cambios de hábito sale de alerta (riesgo {r['risk0']:.0%} → {r['risk']:.0%} "
                "en el mejor caso): la prioridad es una regeneración asistida en el concesionario.")
    parts = []
    if r["long_trips"]:
        parts.append(f"{r['long_trips']} trayecto{'s' if r['long_trips'] > 1 else ''} de ruta de ~30 min por semana")
    if r["short_cut"]:
        parts.append(f"reducir un {r['short_cut']}% los viajes de menos de 5 km (agrupar trámites en un solo viaje)")
    if r["warm"]:
        parts.append("no apagar el motor mientras el tablero indica limpieza del filtro")
    return "Cambio mínimo que saca al vehículo de alerta: " + "; ".join(parts) + \
        f" (riesgo {r['risk0']:.0%} → {r['risk']:.0%})."


def regen_windows(trips, day, weeks=WEEKS, min_mins=MIN_MINS):
    """(P, N)[día, franja]: fracción de semanas con un viaje apto / casi apto (10 a min_mins) antes de `day`."""
    end = pd.Timestamp(day) + pd.Timedelta(days=1)  # incluye el día de análisis completo
    start = end - pd.Timedelta(weeks=weeks)
    t = trips[(trips["lts"] >= start) & (trips["lts"] < end)]
    blk = pd.cut(t["lts"].dt.hour, [b[0] for b in BLOCKS.values()] + [24], right=False, labels=list(BLOCKS))
    week = (t["lts"] - start).dt.days // 7
    out = []
    for sel in (t["mins"] >= min_mins, (t["mins"] >= 10) & (t["mins"] < min_mins)):
        g = pd.DataFrame({"d": t["lts"].dt.dayofweek[sel], "b": blk[sel], "w": week[sel]}).drop_duplicates()
        m = g.groupby(["d", "b"], observed=False).size().unstack(fill_value=0) / weeks
        out.append(m.reindex(index=range(7), columns=list(BLOCKS), fill_value=0).set_axis(DAYS))
    return out[0], out[1]


def window_text(P, N):
    best = P.stack().sort_values(ascending=False)
    (d, b), p = best.index[0], best.iloc[0]
    if p >= 0.5:
        return (f"Los {d} a la {b} hace un trayecto de {MIN_MINS}+ min en {p:.0%} de las semanas: es la mejor "
                f"oportunidad para que termine una regeneración. Si recibe una alerta, que no acorte ese viaje.")
    near = N.stack().sort_values(ascending=False)
    (d2, b2), q = near.index[0], near.iloc[0]
    if q >= 0.5:
        return (f"No tiene un trayecto de {MIN_MINS}+ min habitual, pero los {d2} a la {b2} suele hacer uno de "
                f"10–{MIN_MINS} min ({q:.0%} de las semanas): estirarlo unos minutos alcanza para completar la "
                "limpieza.")
    return (f"No tiene un trayecto de {MIN_MINS}+ min habitual (el más frecuente: {d} a la {b}, {p:.0%} de las "
            f"semanas). Conviene planificar uno por semana.")


def completion(trips, bins=(0, 5, 10, 15, 20, 30, 60, 1e4)):
    """% de regeneraciones en curso al empezar el viaje que terminan antes de apagar el motor, por duración."""
    c = trips[trips["dpf_state0"].str.startswith("Cleaning", na=False)]
    done = ~c["dpf_state1"].str.startswith("Cleaning", na=False)
    return done.groupby(pd.cut(c["mins"], list(bins)), observed=True).agg(["mean", "size"])


def _check():
    # ventana: un vehículo sintético que todos los jueves a las 18:30 hace 35 min y los lunes 8:00 hace 12 min
    day = pd.Timestamp("2026-03-01")  # domingo
    rows = []
    for w in range(WEEKS):
        thu = day - pd.Timedelta(days=3 + 7 * w) + pd.Timedelta(hours=18.5)
        mon = day - pd.Timedelta(days=6 + 7 * w) + pd.Timedelta(hours=8)
        rows += [{"lts": thu, "mins": 35.0}, {"lts": mon, "mins": 12.0},
                 {"lts": mon + pd.Timedelta(hours=2), "mins": 3.0}]
    rows.append({"lts": day + pd.Timedelta(days=2), "mins": 300.0})  # futuro: no debe contar
    P, N = regen_windows(pd.DataFrame(rows), day)
    assert P.loc["jueves", "noche"] == 1 and P.values.sum() == 1, P
    assert N.loc["lunes", "mañana"] == 1 and N.values.sum() == 1, N
    assert "jueves a la noche" in window_text(P, N)
    assert "lunes a la mañana" in window_text(P * 0, N)

    # receta: con el GBM monótono real, más cambio nunca sube el riesgo y la receta respeta el objetivo
    import lightgbm as lgb
    from src.features import feature_cols
    f = pd.read_parquet("data/features.parquet")
    m = lgb.Booster(model_file="models/gbm90_whatif.txt")
    cols = feature_cols(f)
    X = f[cols].sample(300, random_state=0)
    p = m.predict(X)
    target = np.quantile(p, 0.5)
    for i in np.argsort(-p)[:20]:
        x = X.iloc[[i]]
        r = [m.predict(simulate(x, lt))[0] for lt in range(4)]
        assert all(b <= a + 1e-9 for a, b in zip(r, r[1:])), r
        rec = min_recipe(m, x, target)
        assert rec["risk"] <= rec["risk0"] + 1e-9
        assert not rec["reaches"] or rec["risk"] <= target
    print("chequeos OK")


def whatif_target(f, pred, rel):
    """Objetivo de la receta: el mismo umbral relativo de flota que la alerta, en escala what-if."""
    from src.evaluate import fleet_threshold
    return fleet_threshold(pd.DataFrame({"day": f["day"].values, "pw": pred}), "pw", rel)


def summary():
    """Primer día en alerta de cada vehículo del holdout: ¿qué receta sale y tiene una ventana habitual?"""
    import json
    import lightgbm as lgb
    from src.features import feature_cols
    f = pd.read_parquet("data/features.parquet")
    f["day"] = pd.to_datetime(f["day"])
    cols = feature_cols(f)
    sc = pd.read_parquet("data/scores.parquet")
    sc["day"] = pd.to_datetime(sc["day"])
    fold = pd.read_parquet("data/preds.parquet", columns=["v", "day", "fold"])
    fold["day"] = pd.to_datetime(fold["day"])
    f = f.merge(sc, on=["v", "day"]).merge(fold, on=["v", "day"])
    m = lgb.Booster(model_file="models/gbm90_whatif.txt")
    rel = json.load(open("data/metrics.json"))["relative_operating_pct"]
    f["target"] = whatif_target(f, m.predict(f[cols]), rel)
    first = f[(f["fold"] == -1) & (f["score_s"] >= f["thr_rel"])].sort_values("day").groupby("v").head(1)
    t = pd.read_parquet("data/trips.parquet")
    t = {v: g for v, g in t[t["v"].isin(first["v"])].groupby("v")}
    rows = []
    for i, r in first.iterrows():
        rec = min_recipe(m, f.loc[[i], cols], r["target"])
        P, N = regen_windows(t[r["v"]], r["day"])
        rows.append({**rec, "v": r["v"], "failed": r["failed"], "window": P.values.max() >= 0.5,
                     "near": P.values.max() < 0.5 and N.values.max() >= 0.5})
    d = pd.DataFrame(rows)
    need = d[~d["already_ok"]]
    print(f"vehículos del holdout que entran en alerta: {len(d)} ({int(d['failed'].sum())} con evento)")
    print(f"  el modelo what-if ya los ve fuera del top de riesgo: {d['already_ok'].mean():.0%}")
    print(f"  de los demás ({len(need)}), la receta alcanza en {need['reaches'].mean():.0%}; "
          f"reducción mediana del riesgo {(1 - need['risk'] / need['risk0']).median():.0%}")
    print("  componentes de la receta: " + ", ".join(
        f"{k} {v:.0%}" for k, v in {"trayectos de ruta": (need["long_trips"] > 0).mean(),
                                    "menos viajes cortos": (need["short_cut"] > 0).mean(),
                                    "no cortar la limpieza": need["warm"].mean()}.items()))
    print(f"  con ventana habitual (trayecto de {MIN_MINS}+ min en >= 50 % de las semanas): {d['window'].mean():.0%}; "
          f"solo casi apta (estirar un viaje de 10-{MIN_MINS} min): {d['near'].mean():.0%}; "
          f"sin ventana: {1 - d['window'].mean() - d['near'].mean():.0%}")
    return d


if __name__ == "__main__":
    _check()
    t = pd.read_parquet("data/trips.parquet")
    print("regeneración en curso que termina dentro del viaje, por duración (min):")
    print(completion(t).round(3).to_string())
    summary()
