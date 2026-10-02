"""LightGBM en vivo: entrena el componente principal (5 folds + holdout) en CPU, muestra lo que genera y da paso al
dashboard completo con el resto de los modelos ya entrenados. Ejecutar: streamlit run app_lgbm.py"""
import json
import time

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import roc_auc_score

from src.evaluate import REL_PCTS, fleet_threshold, group_of, leadtime, pick_operating, smooth
from src.features import HORIZONS, feature_cols
from src.models import K, fit_gbm, split
from src.ui import detection_chart, risk_chart, weight_chart

st.set_page_config("DPF Health Copilot", layout="wide")
DEMO = "VEH_0543"  # la unidad de Ibagué que sigue el pitch


@st.cache_resource
def features():
    f = pd.read_parquet("data/features.parquet").sort_values(["v", "day"]).reset_index(drop=True)
    cal = pd.read_parquet("data/calendar.parquet", columns=["v", "day", "km"])
    cal["cum_km"] = cal.groupby("v")["km"].cumsum()  # para la anticipación en km (no es feature)
    cum_km = cal.set_index(["v", "day"])["cum_km"].reindex(pd.MultiIndex.from_frame(f[["v", "day"]])).values
    return f, cum_km


def train(f, cum_km, log):
    """Mismo protocolo que src.models: OOF en los 5 folds de train y holdout = promedio de los 5 modelos."""
    cols = feature_cols(f)
    fold = split(f)
    test = np.where(fold == -1)[0]
    P = f[["v", "day", "failed", "tte"] + [f"{p}{h}" for p in ("y", "m") for h in HORIZONS]].copy()
    P["fold"], P["cum_km"] = fold, cum_km
    bar, t0, n = st.progress(0.0), time.time(), 0
    for k in range(K):
        tr, va = np.where((fold != k) & (fold != -1))[0], np.where(fold == k)[0]
        for h in HORIZONS:
            r = tr[f[f"m{h}"].values[tr]]
            m = fit_gbm(f.loc[r, cols], f.loc[r, f"y{h}"])
            P.loc[va, f"gbm{h}"] = m.predict_proba(f.loc[va, cols])[:, 1]
            P.loc[test, f"gbm{h}"] = P.loc[test, f"gbm{h}"].fillna(0) + m.predict_proba(f.loc[test, cols])[:, 1] / K
            n += 1
            bar.progress(n / (K * len(HORIZONS)),
                         f"fold {k + 1}/{K} · {h} días · {len(r):,} filas · {time.time() - t0:.0f} s".replace(",", "."))
    log.write(f"{K * len(HORIZONS)} modelos entrenados en {time.time() - t0:.0f} s")
    return P, m, cols, time.time() - t0  # m: modelo a 90 días del último fold (para SHAP)


def evaluate(P, f, m90, cols, log):
    test, oof = (P["fold"] == -1).values, (P["fold"] >= 0).values
    disc = [{"H": h, "auc": roc_auc_score(P.loc[test & P[f"m{h}"].values, f"y{h}"],
                                          P.loc[test & P[f"m{h}"].values, f"gbm{h}"])} for h in HORIZONS]
    log.write("AUC en los 198 vehículos que el modelo nunca vio")

    # política de alerta del dashboard: top X % de riesgo de la flota (30 días), X elegido en OOF contra la ECU
    P["score_s"] = smooth(P, "gbm90")
    P["ecu"] = (f["w7_sh_over"].fillna(0) > 0).astype(float).values
    rel = {pct: (P["score_s"].values >= fleet_threshold(P, "score_s", pct)).astype(float) for pct in REL_PCTS}
    ecu_oof = leadtime(P[oof], "ecu", 0.5)[1]
    rop = pick_operating([{"pct": pct, **leadtime(P[oof].assign(a=rel[pct][oof]), "a", 0.5)[1]} for pct in REL_PCTS],
                         ecu_oof)
    curve = [{"pct": pct, **leadtime(P[test].assign(a=rel[pct][test]), "a", 0.5)[1]} for pct in REL_PCTS]
    ecu = leadtime(P[test], "ecu", 0.5)[1]
    P["alerta"] = rel[rop].astype(bool)
    P["thr"] = fleet_threshold(P, "score_s", rop)
    log.write(f"Política de alerta: top {rop:.0%} de la flota (elegido fuera de fold)")

    sample = f.loc[test].sample(3000, random_state=0)[cols]
    sv = shap.TreeExplainer(m90.booster_).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    sg = pd.DataFrame({"feature": cols, "shap": np.abs(sv).mean(0)})
    sg["group"] = sg["feature"].map(group_of)
    log.write("SHAP sobre 3000 días de holdout")
    return {"disc": disc, "curve": curve, "ecu": ecu, "rop": rop, "op": next(c for c in curve if c["pct"] == rop),
            "shap": sg.sort_values("shap", ascending=False)}


def live():
    st.title("LightGBM en vivo")
    st.caption("Entrena ahora mismo, en CPU, el modelo principal del ensamble y lo evalúa en 198 vehículos que nunca ve.")
    f, cum_km = features()
    c = st.columns(3)
    c[0].metric("Días de vehículo", f"{len(f):,}".replace(",", "."))
    c[1].metric("Vehículos", f"{f['v'].nunique()}")
    c[2].metric("Vehículos con evento", f"{f.groupby('v')['failed'].first().sum()}")

    if st.button("Volver a entrenar" if "live" in st.session_state else "Entrenar LightGBM", type="primary"):
        with st.status("Entrenando…", expanded=True) as s:
            P, m90, cols, secs = train(f, cum_km, s)
            s.update(label="Evaluando…")
            R = evaluate(P, f, m90, cols, s)
            s.update(label=f"Listo en {secs:.0f} s de entrenamiento", state="complete", expanded=False)
        st.session_state["live"] = (P, R, secs)
    if "live" not in st.session_state:
        return
    P, R, secs = st.session_state["live"]
    test = P[P["fold"] == -1]
    op, ecu = R["op"], R["ecu"]

    c = st.columns(3)
    c[0].metric("Tiempo de entrenamiento", f"{secs:.0f} s", f"{K * len(HORIZONS)} modelos", delta_color="off",
                delta_arrow="off")
    c[1].metric("Eventos anticipados", f"{op['detection_rate']:.0%}", f"ECU: {ecu['detection_rate']:.0%}",
                delta_color="off", delta_arrow="off")
    c[2].metric("Falsas alarmas por vehículo y año", f"{op['false_alarm_episodes_per_vehicle_year']:.2f}"
                .replace(".", ","), f"ECU: {ecu['false_alarm_episodes_per_vehicle_year']:.2f}".replace(".", ","),
                delta_color="off", delta_arrow="off")

    off = {r["H"]: r["auc"] for r in json.load(open("data/metrics.json"))["discrimination"] if r["model"] == "gbm"}
    auc = lambda x: f"{x:.3f}".replace(".", ",")
    st.dataframe(pd.DataFrame([{"Horizonte": f"{r['H']} días", "AUC de esta corrida": auc(r["auc"]),
                                "AUC de la corrida oficial": auc(off[r["H"]])} for r in R["disc"]]), hide_index=True)

    l, r = st.columns(2)
    with l:
        st.subheader("Más eventos detectados con las mismas falsas alarmas")
        st.plotly_chart(detection_chart(R["curve"], ecu, op), width="stretch")
    with r:
        st.subheader("Qué pesa en el riesgo")
        st.plotly_chart(weight_chart(R["shap"].groupby("group")["shap"].sum()), width="stretch")

    st.subheader("Riesgo de un vehículo")
    ev = test[test["tte"].notna()]
    veh = ev["v"].unique().tolist()
    v = st.selectbox("Vehículo con evento", veh, index=veh.index(DEMO) if DEMO in veh else 0)
    g = test[test["v"] == v]
    events = sorted(set(ev.loc[ev["v"] == v, "day"] + pd.to_timedelta(ev.loc[ev["v"] == v, "tte"], "D")))
    st.plotly_chart(risk_chart(g["day"], g["score_s"], g["thr"], g.loc[g["ecu"] > 0, "day"], events), width="stretch")

    with st.expander("Ver las predicciones generadas"):
        last = test.groupby("v").tail(1).sort_values("score_s", ascending=False)
        st.dataframe(last[["v", "score_s", "alerta", "failed"]].rename(columns={
            "v": "Vehículo", "score_s": "Riesgo a 90 días", "alerta": "En alerta", "failed": "Tuvo evento"}),
            hide_index=True, width="stretch",
            column_config={"Riesgo a 90 días": st.column_config.NumberColumn(format="percent")})

    st.page_link("app.py", label="Ver el dashboard completo", icon=":material/arrow_forward:")


st.navigation([st.Page(live, title="LightGBM en vivo", default=True),
               st.Page("app.py", title="Dashboard completo")]).run()
