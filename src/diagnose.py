"""Diagnóstico: ¿el techo de desempeño lo ponen los datos?

  learning  Curva de aprendizaje: LightGBM y la red con 25/50/75/100 % de los vehículos de train, submuestreados por
            vehículo (estratificado por fallado) dentro de cada fold; AUC/AP OOF sobre los folds de validación completos.
  temporal  Descompone la caída del holdout por vehículo (~0.78) a la validación temporal (~0.64), con el LightGBM y los
            vehículos del holdout: período de entrenamiento (todo / etiquetas conocidas en T) x período evaluado
            (< T / >= T), más dos controles de tamaño con datos de todo el período: (a) mismas filas y positivos que
            "antes de T" y (b) además la misma cantidad de vehículos con evento (lo que importa según la curva).
  retrain   Despliegue simulado desde T con reentrenos: estático, trimestral y mensual (etiquetas conocidas a cada fecha).
  drift     Origen de la deriva: validación adversarial (¿se distinguen los días antes/después de T?), ablación de las
            features que más cambian y evolución temporal de eventos y tasas.
  ecu       Política combinada (alerta = modelo O advertencia ECU) contra el modelo solo, en holdout y temporal.
  recency   Pesos exp(-antigüedad / tau) en el entrenamiento del despliegue en T; tau elegido antes de T.

Salidas: data/learning_curve.json, data/temporal_decomp.json, data/retrain.json, data/drift.json,
data/ecu_combo.json, data/recency.json
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
        # control (b): además, tantos vehículos con evento como "antes de T" (el resto de los positivos se descarta)
        veh = f["v"].values
        pos_veh_pre = np.unique(veh[train["antes_de_T"] & (y == 1)])
        pos_veh_all = np.unique(veh[train["todo"] & (y == 1)])
        ctrl = []
        for rep in range(reps):
            rng = np.random.default_rng(100 + rep)
            keep = rng.choice(pos_veh_all, len(pos_veh_pre), replace=False)
            ok = train["todo"] & (~np.isin(veh, pos_veh_all) | np.isin(veh, keep))
            ip, ineg = np.where(ok & (y == 1))[0], np.where(ok & (y == 0))[0]
            r = np.concatenate([ip, rng.choice(ineg, min(len(ineg), n_pre - len(ip)), replace=False)])
            ctrl.append(fit_gbm(f.loc[r, cols], y[r]).predict_proba(f[cols])[:, 1])
        preds["todo_mismos_eventos"] = np.mean(ctrl, 0)
        out = {"train_rows": {"todo": int(train["todo"].sum()), "antes_de_T": n_pre},
               "train_pos": {"todo": int(y[train["todo"]].sum()), "antes_de_T": pos_pre},
               "train_event_vehicles": {"todo": len(pos_veh_all), "antes_de_T": len(pos_veh_pre)}}
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
            out[f"eval_{per}"]["delta_antes_de_T_vs_mismos_eventos"] = paired_ci(
                yt, preds["todo_mismos_eventos"][te], preds["antes_de_T"][te], v)
        res[f"H{h}"] = out
        print(f"H{h}: train todo {out['train_rows']['todo']} filas / {out['train_pos']['todo']} pos / "
              f"{len(pos_veh_all)} vehículos con evento; antes de T {n_pre} / {pos_pre} / {len(pos_veh_pre)}", flush=True)
        for per in ("antes_de_T", "desde_T", "todo"):
            e = out[f"eval_{per}"]
            print(f"  evalúa {per:10s} ({e['n_vehicles']} veh, {e['n_event_vehicles']} con evento, base {e['base_rate']:.3f}): "
                  + " | ".join(f"{n} {e[n]['auc']:.3f} {np.round(e[n]['auc_ci95'], 3)}" for n in preds)
                  + f" | Δ antes_de_T {e['delta_antes_de_T_vs_todo'][0]:+.3f} {e['delta_antes_de_T_vs_todo'][1]}"
                  + f" | Δ submuestreado {e['delta_submuestreado_vs_todo'][0]:+.3f} {e['delta_submuestreado_vs_todo'][1]}"
                  + f" | Δ antes_de_T vs mismos_eventos {e['delta_antes_de_T_vs_mismos_eventos'][0]:+.3f} "
                  + f"{e['delta_antes_de_T_vs_mismos_eventos'][1]}",
                  flush=True)
    json.dump(res, open("data/temporal_decomp.json", "w"), indent=2, default=float)


# ---------------- 3. frecuencia de reentrenamiento ----------------
def retrain(policies=(("estatico", None), ("trimestral", "QS"), ("mensual", "MS"))):
    """En cada fecha de reentreno t se ajusta con las etiquetas conocidas en t y se predice hasta el próximo reentreno."""
    f = load()
    cols = feature_cols(f)
    end = f["day"].max() + pd.Timedelta(days=1)
    every = np.ones(len(f), bool)
    scen = {"vehiculos_nuevos": ((f["fold"] >= 0).values, (f["fold"] == -1).values), "misma_flota": (every, every)}
    res = {"T": str(T.date())}
    for sc, (trs, tes) in scen.items():
        res[sc] = {}
        for h in HORIZONS:
            m, y = f[f"m{h}"].values, f[f"y{h}"].values
            preds = {}
            for name, freq in policies:
                dates = list(pd.date_range(T, end, freq=freq)) if freq else [T]
                p = np.full(len(f), np.nan)
                for i, t in enumerate(dates):
                    nxt = dates[i + 1] if i + 1 < len(dates) else end
                    rows = trs & known_at(f, h, t).values
                    sel = tes & (f["day"] >= t).values & (f["day"] < nxt).values
                    p[sel] = fit_gbm(f.loc[rows, cols], y[rows]).predict_proba(f.loc[sel, cols])[:, 1]
                preds[name] = p
            te = tes & (f["day"] >= T).values & m
            yt, v = y[te], f["v"].values[te]
            out = {}
            for name, p in preds.items():
                ci, _ = cluster_ci(yt, p[te], v, n=200)
                out[name] = {"auc": roc_auc_score(yt, p[te]), "auc_ci95": ci}
                if name != "estatico":
                    out[name]["delta_vs_estatico"] = paired_ci(yt, preds["estatico"][te], p[te], v)
            res[sc][f"H{h}"] = out
            print(f"{sc} H{h}: " + " | ".join(
                f"{n} {o['auc']:.3f} {np.round(o['auc_ci95'], 3)}"
                + (f" Δ {o['delta_vs_estatico'][0]:+.3f} {o['delta_vs_estatico'][1]}" if "delta_vs_estatico" in o else "")
                for n, o in out.items()), flush=True)
    json.dump(res, open("data/retrain.json", "w"), indent=2, default=float)


# ---------------- 4. origen de la deriva ----------------
def drift(top_k=(5, 10, 20)):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    f = load()
    cols = feature_cols(f)
    post = (f["day"] >= T).values
    res = {"T": str(T.date())}
    # (a) validación adversarial: ¿un modelo distingue días antes / después de T? CV agrupada por vehículo
    oof, imp = np.zeros(len(f)), pd.Series(0.0, index=cols)
    for tr, te in GroupKFold(5).split(f, post, f["v"]):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, verbose=-1, random_state=0).fit(f.iloc[tr][cols], post[tr])
        oof[te] = m.predict_proba(f.iloc[te][cols])[:, 1]
        imp += pd.Series(m.booster_.feature_importance("gain"), cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    res["adversarial_auc"] = roc_auc_score(post, oof)
    res["adversarial_top"] = imp.head(20).round(4).to_dict()
    print(f"validación adversarial antes/después de T: AUC {res['adversarial_auc']:.3f}")
    print("features que más separan los períodos:\n" + imp.head(20).round(3).to_string(), flush=True)
    # (b) ablación: modelo de despliegue (etiquetas conocidas en T, vehículos de train) sin las features que más derivan
    trv, tev = (f["fold"] >= 0).values, (f["fold"] == -1).values
    res["ablation"] = {}
    for h in HORIZONS:
        m, y = f[f"m{h}"].values, f[f"y{h}"].values
        rows = trv & known_at(f, h, T).values
        r = {}
        for k in (0,) + tuple(top_k):
            use = [c for c in cols if c not in set(imp.index[:k])]
            p = fit_gbm(f.loc[rows, use], y[rows]).predict_proba(f[use])[:, 1]
            r[f"sin_top{k}"] = {per: roc_auc_score(y[tev & m & sel], p[tev & m & sel])
                                for per, sel in (("antes_de_T", ~post), ("desde_T", post))}
        res["ablation"][f"H{h}"] = r
        print(f"ablación H{h} (AUC holdout antes / desde T): " + " | ".join(
            f"{k} {v['antes_de_T']:.3f} / {v['desde_T']:.3f}" for k, v in r.items()), flush=True)
    # (c) etiquetas en el tiempo: eventos por trimestre y tasa por vehículo activo; tasa de positivos y90 por trimestre
    s = pd.read_parquet("data/static.parquet")
    ev = pd.Series(pd.to_datetime(np.concatenate([np.array(e, dtype="datetime64[D]") for e in s["events"] if len(e)])))
    q = f["day"].dt.to_period("Q")
    tl = pd.DataFrame({"eventos": ev.dt.to_period("Q").value_counts(), "vehiculos_activos": f.groupby(q)["v"].nunique(),
                       "y90_rate": f[f["m90"]].groupby(q[f["m90"]])["y90"].mean()}).sort_index()
    tl["eventos_por_100_vehiculos"] = 100 * tl["eventos"] / tl["vehiculos_activos"]
    res["timeline"] = {str(k): v for k, v in tl.round(4).to_dict("index").items()}
    print(tl.round(3).to_string())
    json.dump(res, open("data/drift.json", "w"), indent=2, default=float)


# ---------------- 5. política combinada: modelo O advertencia ECU ----------------
COMBO_PCTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30)


def _policies(p, ref, ecu_col="ecu"):
    """Alertas de cada política sobre el riesgo suavizado `s`: fija (cuantil de `ref`) y relativa, solas y O ECU."""
    from src.evaluate import FPRS, fleet_threshold
    ecu = p[ecu_col].values > 0
    out = {}
    for x in FPRS:
        a = p["s"].values >= ref.quantile(1 - x)
        out[("fijo", "modelo", x)], out[("fijo", "modelo_o_ecu", x)] = a, a | ecu
    for x in COMBO_PCTS:
        a = p["s"].values >= fleet_threshold(p, "s", x)
        out[("relativo", "modelo", x)], out[("relativo", "modelo_o_ecu", x)] = a, a | ecu
    return out


def _compare(cal, ev, ref, label, cal_mask=None, ev_mask=None):
    """Elige el punto de operación de cada política en `cal` (falsas alarmas <= ECU) y la evalúa en `ev`. Con máscaras,
    las alertas se calculan sobre el frame entero (el umbral relativo mira a toda la flota) y después se recortan."""
    from src.evaluate import leadtime
    A_cal, A_ev = _policies(cal, ref), (_policies(ev, ref) if ev is not cal else None)
    A_ev = A_ev or A_cal
    if cal_mask is not None:
        A_cal = {k: a[cal_mask] for k, a in A_cal.items()}
        cal = cal[cal_mask].reset_index(drop=True)
    if ev_mask is not None:
        A_ev = {k: a[ev_mask] for k, a in A_ev.items()}
        ev = ev[ev_mask].reset_index(drop=True)
    ecu_cal, ecu_ev = leadtime(cal, "ecu", 0.5)[1], leadtime(ev, "ecu", 0.5)
    res, det = {"ecu": ecu_ev[1]}, {"ecu": ecu_ev[0].set_index(["v", "event"])["detected"]}
    for kind in ("fijo", "relativo"):
        for pol in ("modelo", "modelo_o_ecu"):
            keys = [k for k in A_cal if k[:2] == (kind, pol)]
            cal_curve = {k[2]: leadtime(cal.assign(a=A_cal[k].astype(float)), "a", 0.5)[1] for k in keys}
            ok = [x for x, c in cal_curve.items() if c["false_alarm_episodes_per_vehicle_year"]
                  <= ecu_cal["false_alarm_episodes_per_vehicle_year"]]
            op = max(ok) if ok else min(cal_curve)
            curve = {}
            for k in keys:
                lt, s = leadtime(ev.assign(a=A_ev[k].astype(float)), "a", 0.5)
                curve[k[2]] = s
                if k[2] == op:
                    det[f"{kind}_{pol}"] = lt.set_index(["v", "event"])["detected"]
            res[f"{kind}_{pol}"] = {"operating": op, "at_op": curve[op], "curve": curve}
    # IC95 de la diferencia de detección (combinada - modelo solo) remuestreando eventos, a su punto de operación
    rng = np.random.default_rng(0)
    for kind in ("fijo", "relativo"):
        a, b = det[f"{kind}_modelo"].astype(float), det[f"{kind}_modelo_o_ecu"].astype(float).reindex(det[f"{kind}_modelo"].index)
        d = [(b.values[i] - a.values[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        res[f"{kind}_delta_deteccion"] = [float(b.mean() - a.mean()), np.percentile(d, [2.5, 97.5]).round(3).tolist()]
    print(f"== {label}  (ECU: detección {ecu_ev[1]['detection_rate']:.3f}, "
          f"FA {ecu_ev[1]['false_alarm_episodes_per_vehicle_year']:.2f}/veh-año, {ecu_ev[1]['n_events']} eventos)")
    for kind in ("fijo", "relativo"):
        for pol in ("modelo", "modelo_o_ecu"):
            r = res[f"{kind}_{pol}"]
            print(f"  {kind:8s} {pol:13s} op={r['operating']}: detección {r['at_op']['detection_rate']:.3f} "
                  f"FA {r['at_op']['false_alarm_episodes_per_vehicle_year']:.2f} "
                  f"anticipación {r['at_op']['median_lead_days']:.0f} d")
        print(f"  {kind}: Δ detección combinada - modelo {res[f'{kind}_delta_deteccion'][0]:+.3f} "
              f"{res[f'{kind}_delta_deteccion'][1]}")
        for pol in ("modelo", "modelo_o_ecu"):  # curva completa: ¿la combinada domina a igual tasa de falsas alarmas?
            print(f"    curva {pol:13s} " + " ".join(
                f"{x}:{c['detection_rate']:.2f}/{c['false_alarm_episodes_per_vehicle_year']:.2f}"
                for x, c in res[f"{kind}_{pol}"]["curve"].items()), flush=True)
    return res


def ecu_combo():
    """Alerta = modelo O advertencia ECU. Holdout por vehículo (stack, punto elegido en OOF) y validación temporal
    (LightGBM, punto elegido en la calibración [T-90, T), como src.temporal)."""
    from src.evaluate import smooth
    from src.temporal import CAL, fit
    res = {}
    # --- holdout por vehículo ---
    p = pd.read_parquet("data/preds.parquet", columns=["v", "day", "fold", "failed", "tte"])
    p["day"] = pd.to_datetime(p["day"])
    sc = pd.read_parquet("data/scores.parquet")
    sc["day"] = pd.to_datetime(sc["day"])
    p = p.merge(sc[["v", "day", "score_s", "ecu_warning"]], on=["v", "day"]).rename(columns={"score_s": "s", "ecu_warning": "ecu"})
    p = p.sort_values(["v", "day"]).reset_index(drop=True)
    oof, test = (p["fold"] >= 0).values, (p["fold"] == -1).values
    res["holdout"] = _compare(p, p, p.loc[oof & (p["failed"] == 0).values, "s"],
                              "holdout por vehículo (punto elegido en OOF)", oof, test)
    # --- validación temporal ---
    f = load().sort_values(["v", "day"]).reset_index(drop=True)
    cols = feature_cols(f)
    every = pd.Series(True, index=f.index)
    t0 = T - pd.Timedelta(days=CAL)
    for name, (trv, tev) in {"vehiculos_nuevos": (f["fold"] >= 0, f["fold"] == -1), "misma_flota": (every, every)}.items():
        inner = fit(f, trv & known_at(f, 90, t0), 90, cols)
        cal = trv & (f["day"] >= t0) & (f["day"] < T)
        c = f.loc[cal, ["v", "day", "tte"]].copy()
        c.loc[c["day"] + pd.to_timedelta(c["tte"], "D") >= T, "tte"] = np.nan  # en T no se conocen eventos futuros
        c["failed"] = c["v"].isin(set(c.loc[c["tte"].notna(), "v"])).astype(int)
        c["p"] = inner.predict_proba(f.loc[cal, cols])[:, 1]
        c["s"] = smooth(c, "p")
        c["ecu"] = (f.loc[cal, "w7_sh_over"].fillna(0) > 0).astype(float)
        final = fit(f, trv & known_at(f, 90, T), 90, cols)
        fut = tev & (f["day"] >= T)
        q = f.loc[fut, ["v", "day", "failed", "tte"]].copy()
        q["p"] = final.predict_proba(f.loc[fut, cols])[:, 1]
        q["s"] = smooth(q, "p")
        q["ecu"] = (f.loc[fut, "w7_sh_over"].fillna(0) > 0).astype(float)
        res[f"temporal_{name}"] = _compare(c.reset_index(drop=True), q.reset_index(drop=True),
                                           c.loc[c["failed"] == 0, "s"], f"temporal, {name} (punto elegido en [T-90, T))")
    json.dump(res, open("data/ecu_combo.json", "w"), indent=2, default=float)


# ---------------- 6. peso a los datos recientes ----------------
def recency(taus=(None, 60, 120, 240, 480), sel_window=180):
    """Despliegue en T con pesos exp(-antigüedad / tau) en el entrenamiento. tau se elige sin mirar el período >= T:
    modelo con lo conocido en T - sel_window, evaluado en [T - sel_window, T) con las etiquetas conocidas en T."""
    f = load()
    cols = feature_cols(f)
    every = np.ones(len(f), bool)
    scen = {"vehiculos_nuevos": ((f["fold"] >= 0).values, (f["fold"] == -1).values), "misma_flota": (every, every)}
    t_sel = T - pd.Timedelta(days=sel_window)
    day = f["day"]
    import lightgbm as lgb
    from src.models import GBM_PARAMS, geo_mono

    def fit_w(rows, y, t, tau):
        w = None if tau is None else np.exp(-(t - day[rows]).dt.days.values / tau)
        return lgb.LGBMClassifier(**GBM_PARAMS, **geo_mono(cols)).fit(f.loc[rows, cols], y[rows], sample_weight=w)

    res = {"T": str(T.date()), "sel_window": sel_window}
    for sc, (trs, tes) in scen.items():
        sel_auc, preds, out = {}, {}, {}
        for h in HORIZONS:
            m, y = f[f"m{h}"].values, f[f"y{h}"].values
            sel_rows = trs & known_at(f, h, t_sel).values
            sel_eval = trs & (day >= t_sel).values & (day < T).values & known_at(f, h, T).values
            fut = tes & (day >= T).values & m
            for tau in taus:
                ps = fit_w(sel_rows, y, t_sel, tau).predict_proba(f.loc[sel_eval, cols])[:, 1]
                sel_auc[(h, tau)] = roc_auc_score(y[sel_eval], ps)
                preds[(h, tau)] = fit_w(trs & known_at(f, h, T).values, y, T, tau).predict_proba(f.loc[fut, cols])[:, 1]
            yt, v = y[fut], f["v"].values[fut]
            out[f"H{h}"] = {}
            for tau in taus:
                o = {"sel_auc": sel_auc[(h, tau)], "auc": roc_auc_score(yt, preds[(h, tau)])}
                if tau is not None:
                    o["delta_vs_sin_peso"] = paired_ci(yt, preds[(h, None)], preds[(h, tau)], v)
                out[f"H{h}"][str(tau)] = o
            print(f"{sc} H{h}: " + " | ".join(
                f"tau={t} sel {o['sel_auc']:.3f} fut {o['auc']:.3f}"
                + (f" Δ {o['delta_vs_sin_peso'][0]:+.3f} {o['delta_vs_sin_peso'][1]}" if "delta_vs_sin_peso" in o else "")
                for t, o in out[f"H{h}"].items()), flush=True)
        best = max(taus, key=lambda t: np.mean([sel_auc[(h, t)] for h in HORIZONS]))
        out["tau_elegido"] = best
        print(f"{sc}: tau elegido en la ventana de selección = {best}", flush=True)
        res[sc] = out
    json.dump(res, open("data/recency.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    {"learning": learning, "retrain": retrain, "drift": drift, "ecu": ecu_combo, "recency": recency}.get(
        next(iter(sys.argv[1:]), ""), temporal_decomp)()
