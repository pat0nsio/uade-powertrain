"""Alerta a bordo con memoria fija (EMA + umbral fijo) vs ECU; `export` genera el C y verifica paridad."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.evaluate import leadtime
from src.models import GBM_PARAMS, K, split
from src.temporal import CAL, T, known_at

HALF_LIVES = (7, 30)  # días
TRIP_RATIOS = ["sh_short5", "sh_micro", "sh_urban", "sh_never_warm", "sh_cold_start", "sh_trip_end_in_regen", "sh_idle"]
MSG_RATIOS = ["sh_over", "sh_full", "sh_overloaded", "acc_mean"]
DEPTHS = (2, 3, 4)
COMPACT = ((50, 7), (100, 7), (200, 7), (100, 15))  # (árboles, hojas) de los LightGBM compactos
H = 90


def ema_base(cal):
    """Las 17 series diarias que promedia la EMA (mismo orden que dpf_day_t en edge/dpf_edge.h, con horas = min/60)."""
    base = pd.DataFrame({"v": cal["v"], "n_trips": cal["n_trips"], "n_msgs": cal["n_msgs"].fillna(0), "km": cal["km"],
                         "hours": cal["mins"] / 60, "n_regen": cal["n_regen"], "n_regen_stopped": cal["n_regen_stopped"]})
    for c in TRIP_RATIOS:
        base[c + "__n"] = (cal[c] * cal["n_trips"]).fillna(0)
    for c in MSG_RATIOS:
        base[c + "__n"] = (cal[c] * cal["n_msgs"]).fillna(0)
    return base


def ema_features(cal):
    """Una fila por vehículo-día calendario. Cada EMA se actualiza una vez por día con O(1) memoria por señal."""
    cal = cal.sort_values(["v", "day"]).reset_index(drop=True)
    base = ema_base(cal)
    out = pd.DataFrame({"v": cal["v"], "day": cal["day"]})
    g = base.drop(columns="v").groupby(base["v"], sort=False)
    for hl in HALF_LIVES:
        e = g.transform(lambda s: s.ewm(halflife=hl, adjust=False).mean())
        p = f"e{hl}_"
        for c in TRIP_RATIOS:
            out[p + c] = e[c + "__n"] / e["n_trips"].replace(0, np.nan)
        for c in MSG_RATIOS:
            out[p + c] = e[c + "__n"] / e["n_msgs"].replace(0, np.nan)
        out[p + "km_per_trip"] = e["km"] / e["n_trips"].replace(0, np.nan)
        out[p + "speed"] = e["km"] / e["hours"].replace(0, np.nan)
        out[p + "km_per_day"] = e["km"]
        out[p + "regen_per_1000km"] = e["n_regen"] / e["km"].replace(0, np.nan) * 1000
        out[p + "regen_stop_ratio"] = e["n_regen_stopped"] / (e["n_regen"] + e["n_regen_stopped"]).replace(0, np.nan)
    for c in [c[3:] for c in out.columns if c.startswith("e7_")]:
        out["trend_" + c] = out["e7_" + c] - out["e30_" + c]
    # contadores desde la última regeneración (la ECU ya los lleva)
    grp = (cal["n_regen"] > 0).groupby(cal["v"]).cumsum()
    out["km_since_regen"] = cal["km"].groupby([cal["v"], grp]).cumsum().where(grp > 0)
    out["days_since_regen"] = cal.groupby([cal["v"], grp]).cumcount().astype(float).where(grp > 0)
    return out


def load():
    f = pd.read_parquet("data/features.parquet", columns=["v", "day", "failed", "tte", "w7_sh_over", "w7_sh_full"] +
                        [f"{x}{h}" for x in ("y", "m") for h in (30, 60, 90)])
    f["day"] = pd.to_datetime(f["day"])
    f["fold"] = split(f)
    cal = pd.read_parquet("data/calendar.parquet")
    cal["day"] = pd.to_datetime(cal["day"])
    e = ema_features(cal)
    f = f.merge(e, on=["v", "day"], how="left")
    f["ecu"] = (f["w7_sh_over"].fillna(0) > 0).astype(float)
    cols = [c for c in e.columns if c not in ("v", "day")]
    return f.sort_values(["v", "day"]).reset_index(drop=True), cols


def models():
    m = {f"arbol_d{d}": (lambda d=d: DecisionTreeClassifier(max_depth=d, min_samples_leaf=2000, class_weight="balanced",
                                                             random_state=0)) for d in DEPTHS}
    m["lgbm_ema"] = lambda: lgb.LGBMClassifier(**GBM_PARAMS)
    for n, leaves in COMPACT:
        m[f"lgbm_{n}x{leaves}"] = lambda n=n, leaves=leaves: lgb.LGBMClassifier(
            **{**GBM_PARAMS, "n_estimators": n, "num_leaves": leaves, "learning_rate": 0.03 * 600 / n})
    return m


def size_kb(m):
    """KB a bordo: 9 B por nodo interno, 4 B por hoja."""
    if isinstance(m, DecisionTreeClassifier):
        n_leaf = m.get_n_leaves()
        return ((m.tree_.node_count - n_leaf) * 9 + n_leaf * 4) / 1024
    d = m.booster_.trees_to_dataframe()
    internal = d["split_feature"].notna().sum()
    return (internal * 9 + (len(d) - internal) * 4) / 1024


def fit(make, X, y):
    m = make()
    return m.fit(X.fillna(-1) if isinstance(m, DecisionTreeClassifier) else X, y)  # -1 = "sin dato" para el árbol


def score(m, X):
    return m.predict_proba(X.fillna(-1) if isinstance(m, DecisionTreeClassifier) else X)[:, 1]


FPR_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)  # % de días sanos en alerta


def not_worse(c, ecu):
    """Ni más episodios de falsa alarma ni más días sanos en alerta que la ECU."""
    return (c["false_alarm_episodes_per_vehicle_year"] <= ecu["false_alarm_episodes_per_vehicle_year"]
            and c["healthy_day_alarm_rate"] <= ecu["healthy_day_alarm_rate"])


def operate(cal, s_cal, ref_cal, ev, s_ev, ref_ev, ecu_cal, ecu_ev):
    """% de días sanos en alerta elegido en `cal`, traducido a umbral con `ref_ev`; más detección a igual FA."""
    ref_cal, ref_ev = pd.Series(ref_cal), pd.Series(ref_ev)
    cv = {x: leadtime(cal.assign(a=s_cal), "a", ref_cal.quantile(1 - x))[1] for x in FPR_GRID}
    ok = [x for x, c in cv.items() if not_worse(c, ecu_cal)]
    op = max(ok, key=lambda x: cv[x]["detection_rate"]) if ok else min(FPR_GRID)
    lt, st = leadtime(ev.assign(a=s_ev), "a", ref_ev.quantile(1 - op))
    ev_curve = {x: leadtime(ev.assign(a=s_ev), "a", ref_ev.quantile(1 - x)) for x in FPR_GRID}
    ok = [x for x, (_, c) in ev_curve.items() if not_worse(c, ecu_ev)]
    m = max(ok, key=lambda x: ev_curve[x][1]["detection_rate"]) if ok else min(FPR_GRID)
    return {"fpr_op": op, "op": st, "lt": lt, "same_fa": ev_curve[m][1], "same_fa_lt": ev_curve[m][0]}


def paired_detection(lt_a, lt_b, n=2000, seed=0):
    """IC95 de detección(b) - detección(a) remuestreando eventos (mismos eventos en las dos políticas)."""
    a = lt_a.set_index(["v", "event"])["detected"].astype(float)
    b = lt_b.set_index(["v", "event"])["detected"].astype(float).reindex(a.index)
    rng = np.random.default_rng(seed)
    d = [(b.values[i] - a.values[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(b.mean() - a.mean()), np.percentile(d, [2.5, 97.5]).round(3).tolist()]


def summarize(o, ecu_lt):
    return {"fpr_op": o["fpr_op"], "op": o["op"], "same_fa": o["same_fa"],
            "delta_op": paired_detection(ecu_lt, o["lt"]), "delta_same_fa": paired_detection(ecu_lt, o["same_fa_lt"])}


def rules(ev, ecu_lt):
    """Reglas fijas sin entrenamiento: avisar un nivel antes que la ECU (mensaje 'Full' en vez de 'Over Limit')."""
    out = {}
    for name, col in (("regla_full_7d", "w7_sh_full"), ("regla_full_30d", "e30_sh_full")):
        lt, st = leadtime(ev.assign(a=(ev[col].fillna(0) > 0).astype(float)), "a", 0.5)
        out[name] = {"op": st, "delta_op": paired_detection(ecu_lt, lt)}
    return out


def holdout(f, cols):
    res, trees = {}, {}
    y, m = f[f"y{H}"].values, f[f"m{H}"].values
    oof, test = f["fold"].values >= 0, f["fold"].values == -1
    healthy = f["failed"].values == 0
    ecu_oof, (ecu_lt, ecu_te) = leadtime(f[oof], "ecu", 0.5)[1], leadtime(f[test], "ecu", 0.5)
    for name, make in models().items():
        s, s_avg = np.full(len(f), np.nan), np.zeros(len(f))
        for k in range(K):
            tr = (f["fold"].values >= 0) & (f["fold"].values != k) & m
            va = f["fold"].values == k
            mk = fit(make, f.loc[tr, cols], y[tr])
            s[va] = score(mk, f.loc[va, cols])
            s_avg[test] += score(mk, f.loc[test, cols]) / K
        if name.startswith("arbol"):  # la regla que se instala es UN árbol: el final, con umbral por % de días sanos
            final = fit(make, f.loc[oof & m, cols], y[oof & m])
            s[test] = score(final, f.loc[test, cols])
            ref_ev = score(final, f.loc[oof & healthy, cols])
            trees[name] = export_text(final, feature_names=cols, show_weights=True, decimals=3)
        else:  # LightGBM: promedio de los modelos de cada fold (misma escala que OOF, como en src.models)
            s[test] = s_avg[test]
            ref_ev = s[oof & healthy]
        o = operate(f[oof], s[oof], s[oof & healthy], f[test], s[test], ref_ev, ecu_oof, ecu_te)
        res[name] = {"auc_oof": roc_auc_score(y[oof & m], s[oof & m]),
                     "auc_holdout": roc_auc_score(y[test & m], s[test & m]), **summarize(o, ecu_lt)}
        res[name]["size_kb"] = size_kb(final if name.startswith("arbol") else mk)
        if name.startswith("arbol"):
            res[name]["n_leaves"] = int(final.get_n_leaves())
            res[name]["features_used"] = sorted({cols[i] for i in final.tree_.feature if i >= 0})
    res.update(rules(f[test], ecu_lt))
    res["ecu"] = ecu_te
    return res, trees


def temporal(f, cols):
    """Despliegue en T; % de días sanos en alerta elegido en [T-90, T) con un modelo entrenado en T-90."""
    res = {}
    t0 = T - pd.Timedelta(days=CAL)
    y = f[f"y{H}"].values
    every = np.ones(len(f), bool)
    for sc, (trv, tev) in {"vehiculos_nuevos": (f["fold"].values >= 0, f["fold"].values == -1),
                           "misma_flota": (every, every)}.items():
        cal = trv & (f["day"] >= t0).values & (f["day"] < T).values
        c = f.loc[cal, ["v", "day", "tte", "ecu"]].copy()
        c.loc[c["day"] + pd.to_timedelta(c["tte"], "D") >= T, "tte"] = np.nan  # en T no se conocen eventos futuros
        c["failed"] = c["v"].isin(set(c.loc[c["tte"].notna(), "v"])).astype(int)
        h_cal = c["failed"].values == 0
        fut = tev & (f["day"] >= T).values
        q = f.loc[fut, ["v", "day", "failed", "tte", "ecu", "w7_sh_full", "e30_sh_full"]]
        ecu_cal, (ecu_lt, ecu_fut) = leadtime(c, "ecu", 0.5)[1], leadtime(q, "ecu", 0.5)
        res[sc] = {"ecu": ecu_fut, **rules(q, ecu_lt)}
        mm = f.loc[fut, f"m{H}"].values
        for name, make in models().items():
            rows_in = trv & known_at(f, H, t0).values
            s_cal = score(fit(make, f.loc[rows_in, cols], y[rows_in]), f.loc[cal, cols])
            rows = trv & known_at(f, H, T).values
            final = fit(make, f.loc[rows, cols], y[rows])
            s = score(final, f.loc[fut, cols])
            ref_ev = score(final, f.loc[cal, cols])[h_cal]
            o = operate(c, s_cal, s_cal[h_cal], q, s, ref_ev, ecu_cal, ecu_fut)
            res[sc][name] = {"auc": roc_auc_score(y[fut][mm], s[mm]), **summarize(o, ecu_lt)}
    return res


def fmt(st):
    return (f"detección {st['detection_rate']:.0%}, FA {st['false_alarm_episodes_per_vehicle_year']:.2f}/veh-año, "
            f"días sanos en alerta {st['healthy_day_alarm_rate']:.1%}, anticipación {st['median_lead_days']:.0f} d")


def show(name, r):
    d = r["delta_op"]
    line = f"  {name:14s} " + (f"AUC {r.get('auc', r.get('auc_holdout')):.3f} | " if "fpr_op" in r else "")
    line += f"operación: {fmt(r['op'])} | Δ vs ECU {d[0]:+.3f} {d[1]}"
    if "size_kb" in r:
        line += f" | {r['size_kb']:.1f} KB"
    if "same_fa" in r:
        e = r["delta_same_fa"]
        line += f"\n  {'':14s} igual FA que la ECU (a posteriori): {fmt(r['same_fa'])} | Δ {e[0]:+.3f} {e[1]}"
    print(line, flush=True)


def main():
    f, cols = load()
    print(f"{len(cols)} señales a bordo (EMA {HALF_LIVES} días + tendencias + contadores)", flush=True)
    R = {"signals": cols}
    R["holdout"], R["trees"] = holdout(f, cols)
    print(f"== holdout ({R['holdout']['ecu']['n_events']} eventos). ECU: {fmt(R['holdout']['ecu'])}")
    for name, r in R["holdout"].items():
        if name != "ecu":
            show(name, r)
            if "n_leaves" in r:
                print(f"  {'':14s} AUC OOF {r['auc_oof']:.3f}; {r['n_leaves']} hojas, señales: {r['features_used']}")
    for name, txt in R["trees"].items():
        print(f"-- {name}\n{txt}")
    R["temporal"] = temporal(f, cols)
    for sc, rr in R["temporal"].items():
        print(f"== temporal {sc} ({rr['ecu']['n_events']} eventos). ECU: {fmt(rr['ecu'])}")
        for name, r in rr.items():
            if name != "ecu":
                show(name, r)
    json.dump(R, open("data/edge.json", "w"), indent=2, default=float)


# ---------------- exportación a C ----------------
EDGE_MODEL = "lgbm_100x7"  # el más chico con el mejor AUC OOF entre los compactos (0.732; ver data/edge.json)
C_DIR = "edge"


def alpha(hl):
    """El alpha que usa pandas para ewm(halflife=hl): misma secuencia de operaciones -> mismo double."""
    com = 1 / (1 - np.exp(np.log(0.5) / hl)) - 1
    return 1.0 / (1.0 + com)


def c_model(booster, thr_raw):
    """dpf_edge_model.h: árboles aplanados (índices globales; hoja = ~índice) con los doubles exactos (hex)."""
    feat, thr, left, right, dleft, miss, leaves, roots = [], [], [], [], [], [], [], []
    depth = 0
    MISS = {"None": 0, "Zero": 1, "NaN": 2}

    def walk(node, d):
        nonlocal depth
        depth = max(depth, d)
        if "leaf_value" in node:
            leaves.append(node["leaf_value"])
            return ~(len(leaves) - 1)
        assert node["decision_type"] == "<="
        i = len(feat)
        feat.append(node["split_feature"]); thr.append(node["threshold"]); dleft.append(int(node["default_left"]))
        miss.append(MISS[node["missing_type"]]); left.append(0); right.append(0)
        left[i] = walk(node["left_child"], d + 1)
        right[i] = walk(node["right_child"], d + 1)
        return i

    for t in booster.dump_model()["tree_info"]:
        roots.append(walk(t["tree_structure"], 0))
    arr = lambda typ, name, xs, f=str: f"static const {typ} {name}[{len(xs)}] = {{{', '.join(f(x) for x in xs)}}};"
    return "\n".join([
        "/* GENERADO por `python -m src.edge export` a partir del modelo entrenado. No editar. No versionar (datos Ford). */",
        "#ifndef DPF_EDGE_MODEL_H", "#define DPF_EDGE_MODEL_H", "#include <stdint.h>", "",
        f"#define DPF_N_TREES {len(roots)}", f"#define DPF_MAX_DEPTH {depth}",
        "#define DPF_MISSING_NONE 0", "#define DPF_MISSING_ZERO 1", "#define DPF_MISSING_NAN 2", "",
        "/* calibración: constantes de las EMA (vidas medias 7 y 30 días) y umbral de alerta en escala logit */",
        arr("double", "DPF_ALPHA", [alpha(h) for h in HALF_LIVES], float.hex),
        f"static const double DPF_THRESHOLD_RAW = {float.hex(float(thr_raw))};", "",
        arr("int32_t", "DPF_TREE_ROOT", roots), arr("uint8_t", "DPF_NODE_FEATURE", feat),
        arr("double", "DPF_NODE_THRESHOLD", thr, float.hex), arr("int32_t", "DPF_NODE_LEFT", left),
        arr("int32_t", "DPF_NODE_RIGHT", right), arr("uint8_t", "DPF_NODE_DEFAULT_LEFT", dleft),
        arr("uint8_t", "DPF_NODE_MISSING", miss), arr("double", "DPF_LEAF_VALUE", leaves, float.hex),
        "", "#endif", ""]), len(feat), len(leaves)


def export(n_test=40):
    """Entrena el modelo compacto, genera edge/dpf_edge_model.h y verifica paridad C vs Python."""
    import subprocess
    f, cols = load()
    R = json.load(open("data/edge.json"))
    fpr = R["holdout"][EDGE_MODEL]["fpr_op"]
    y, m = f[f"y{H}"].values, f[f"m{H}"].values
    oof, test = f["fold"].values >= 0, f["fold"].values == -1
    model = fit(models()[EDGE_MODEL], f.loc[oof & m, cols], y[oof & m])
    b = model.booster_
    raw = b.predict(f[cols], raw_score=True)
    # umbral con puntajes fuera de fold (en su propio train los días sanos puntúan más bajo -> umbral demasiado bajo)
    raw_oof = np.full(len(f), np.nan)
    for k in range(K):
        tr = oof & (f["fold"].values != k) & m
        va = f["fold"].values == k
        raw_oof[va] = fit(models()[EDGE_MODEL], f.loc[tr, cols], y[tr]).booster_.predict(f.loc[va, cols], raw_score=True)
    thr_raw = float(np.quantile(raw_oof[oof & (f["failed"].values == 0)], 1 - fpr))
    ecu_lt, ecu = leadtime(f[test], "ecu", 0.5)
    lt, st = leadtime(f[test].assign(a=raw[test]), "a", thr_raw)
    print(f"modelo exportado ({EDGE_MODEL}, {fpr:.1%} de días sanos en alerta elegido en OOF), holdout:")
    print(f"  ECU:    {fmt(ecu)}\n  a bordo: {fmt(st)} | Δ detección {paired_detection(ecu_lt, lt)}")

    src, n_nodes, n_leaves = c_model(b, thr_raw)
    open(f"{C_DIR}/dpf_edge_model.h", "w").write(src)
    kb = (n_nodes * (1 + 8 + 4 + 4 + 1 + 1) + n_leaves * 8) / 1024
    print(f"  {len(b.dump_model()['tree_info'])} árboles, {n_nodes} nodos, {n_leaves} hojas: ~{kb:.1f} KB de tablas en "
          f"double; estado {DPF_STATE_BYTES} B por vehículo")

    # paridad: días calendario de vehículos del holdout (mitad con evento), mismas entradas que dpf_day_t
    cal = pd.read_parquet("data/calendar.parquet")
    cal["day"] = pd.to_datetime(cal["day"])
    cal = cal.sort_values(["v", "day"]).reset_index(drop=True)
    ho = f.loc[test].groupby("v")["failed"].first()
    vs = list(ho[ho == 1].index[: n_test // 2]) + list(ho[ho == 0].index[: n_test // 2])
    c = cal[cal["v"].isin(vs)].reset_index(drop=True)
    base = ema_base(c)
    base["hours"] = c["mins"]  # dpf_day_t recibe minutos; el C divide por 60 igual que pandas
    lines = "".join(f"{v}," + ",".join(f"{x:.17g}" for x in row) + "\n"
                    for v, row in zip(base["v"], base.drop(columns="v").to_numpy(float)))
    exe = "data/dpf_edge_test"  # binario de prueba fuera del repo
    subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror", "-Iedge",
                    f"{C_DIR}/dpf_edge.c", f"{C_DIR}/test_dpf_edge.c", "-lm", "-o", exe], check=True)
    out = subprocess.run([exe], input=lines, capture_output=True, text=True, check=True).stdout
    got = pd.read_csv(pd.io.common.StringIO(out), header=None, names=["v", "raw", "alert"])
    e = ema_features(c)
    exp = b.predict(e[cols], raw_score=True)
    diff = np.abs(got["raw"].values - exp)
    same_alert = (got["alert"].values == (exp >= thr_raw)).mean()
    print(f"paridad C vs Python: {len(c)} días de {len(vs)} vehículos, |Δ puntaje| máx {diff.max():.2e}, "
          f"alertas idénticas {same_alert:.2%}")
    assert diff.max() < 1e-9 and same_alert == 1.0
    return st


DPF_STATE_BYTES = (len(HALF_LIVES) * 17 + 2) * 8 + 2


if __name__ == "__main__":
    import sys
    export() if "export" in sys.argv else main()
