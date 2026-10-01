"""LightGBM en vivo: entrena el componente principal (5 folds + holdout) en CPU, muestra lo que genera y da paso al
dashboard completo con el resto de los modelos ya entrenados. Ejecutar: streamlit run app_lgbm.py"""
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import REL_PCTS, cluster_ci, fleet_threshold, group_of, leadtime, pick_operating, smooth
from src.features import HORIZONS, feature_cols
from src.models import K, fit_gbm, split
from src.ui import AQUA, BLUE, GRAY, INK2, ORANGE, RED, style

st.set_page_config("DPF Health Copilot", layout="wide")


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
    disc = []
    for h in HORIZONS:
        mt, mo = test & P[f"m{h}"].values, oof & P[f"m{h}"].values
        y, s = P.loc[mt, f"y{h}"], P.loc[mt, f"gbm{h}"]
        auc_ci, ap_ci = cluster_ci(y, s, P.loc[mt, "v"])
        disc.append({"H": h, "auc": roc_auc_score(y, s), "auc_ci95": auc_ci, "ap": average_precision_score(y, s),
                     "ap_ci95": ap_ci, "base": y.mean(),
                     "oof_auc": roc_auc_score(P.loc[mo, f"y{h}"], P.loc[mo, f"gbm{h}"])})
    log.write("AUC en holdout con IC por vehículo")

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
    st.caption("Entrena el componente principal del ensamble ahora mismo, en CPU: 5 folds × 3 horizontes (30/60/90 "
               "días) con los vehículos de train, y evalúa en 198 vehículos que el modelo nunca ve.")
    f, cum_km = features()
    c = st.columns(4)
    c[0].metric("Vehículo-días", f"{len(f):,}".replace(",", "."))
    c[1].metric("Vehículos", f"{f['v'].nunique()}")
    c[2].metric("Features", f"{len(feature_cols(f))}")
    c[3].metric("Vehículos con evento", f"{f.groupby('v')['failed'].first().sum()}")

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
    d90, op, ecu = R["disc"][-1], R["op"], R["ecu"]

    c = st.columns(4)
    c[0].metric("Tiempo de entrenamiento", f"{secs:.0f} s", f"{K * len(HORIZONS)} modelos", delta_color="off")
    c[1].metric("AUC 90 días (holdout)", f"{d90['auc']:.3f}", f"IC95 {d90['auc_ci95'][0]:.2f}–{d90['auc_ci95'][1]:.2f}",
                delta_color="off")
    c[2].metric("Eventos detectados antes de ocurrir", f"{op['detection_rate']:.0%}",
                f"{(op['detection_rate'] - ecu['detection_rate']) * 100:+.0f} pts vs ECU ({ecu['detection_rate']:.0%})")
    c[3].metric("Falsas alarmas / vehículo-año", f"{op['false_alarm_episodes_per_vehicle_year']:.2f}",
                f"ECU {ecu['false_alarm_episodes_per_vehicle_year']:.2f}", delta_color="off")

    st.subheader("Discriminación por horizonte")
    off = {r["H"]: r["auc"] for r in json.load(open("data/metrics.json"))["discrimination"] if r["model"] == "gbm"}
    st.dataframe(pd.DataFrame([{
        "Horizonte (días)": r["H"],
        "AUC holdout (IC95 por vehículo)": f"{r['auc']:.3f} [{r['auc_ci95'][0]:.3f}–{r['auc_ci95'][1]:.3f}]",
        "AP holdout (IC95)": f"{r['ap']:.3f} [{r['ap_ci95'][0]:.3f}–{r['ap_ci95'][1]:.3f}]",
        "Tasa base": f"{r['base']:.3f}",
        "AUC fuera de fold (train)": f"{r['oof_auc']:.3f}",
        "AUC de la corrida oficial": f"{off[r['H']]:.3f}"} for r in R["disc"]]), hide_index=True, width="stretch")
    st.caption("La última columna es el LightGBM de `data/metrics.json`: la corrida en vivo debe reproducirla.")

    l, r = st.columns(2)
    with l:
        st.subheader("Detección vs falsas alarmas")
        st.caption(f"Alerta = vehículo en el top X % de riesgo de la flota (últimos 30 días). X = {R['rop']:.0%}, "
                   "elegido fuera de fold con no más falsas alarmas que la ECU, sin mirar el holdout.")
        lc = pd.DataFrame(R["curve"])
        fig = go.Figure()
        fig.add_scatter(x=lc["false_alarm_episodes_per_vehicle_year"], y=lc["detection_rate"], mode="lines+markers",
                        name="LightGBM", line=dict(color=BLUE, width=2), marker=dict(size=8), customdata=lc["pct"],
                        hovertemplate="top %{customdata:.0%}<br>falsas alarmas=%{x:.2f}<br>detección=%{y:.0%}"
                                      "<extra></extra>")
        fig.add_scatter(x=[ecu["false_alarm_episodes_per_vehicle_year"]], y=[ecu["detection_rate"]],
                        mode="markers+text", name="Advertencia ECU actual", text=["ECU"], textposition="bottom right",
                        marker=dict(color=ORANGE, size=11, symbol="diamond"), hoverinfo="skip")
        fig.add_scatter(x=[op["false_alarm_episodes_per_vehicle_year"]], y=[op["detection_rate"]], mode="markers+text",
                        name="Punto de operación", text=[f"{op['detection_rate']:.0%}"], textposition="top left",
                        marker=dict(color=AQUA, size=14, symbol="circle-open", line_width=3), hoverinfo="skip")
        st.plotly_chart(style(fig, 380, hovermode="closest", xaxis_title="episodios de falsa alarma por vehículo-año",
                              yaxis_title="eventos detectados antes de ocurrir", yaxis_tickformat=".0%"),
                        width="stretch")
    with r:
        st.subheader("Qué explica el riesgo (SHAP, 90 días)")
        g = R["shap"].groupby("group")["shap"].sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 380, hovermode="closest", xaxis_title="Σ |SHAP| medio"), width="stretch")

    st.subheader("Riesgo de un vehículo del holdout")
    ev = test[test["tte"].notna()]
    veh = ev["v"].unique().tolist()
    v = st.selectbox(f"Vehículos del holdout con evento ({len(veh)})", veh)
    g = test[test["v"] == v]
    fig = go.Figure()
    fig.add_scatter(x=g["day"], y=g["thr"], name="Umbral de la flota", line=dict(color=GRAY, dash="dash", width=1))
    fig.add_scatter(x=g["day"], y=g["score_s"], name="Riesgo 90 días (media 7 d)", line=dict(color=BLUE, width=2))
    a = g[g["alerta"]]
    fig.add_scatter(x=a["day"], y=a["score_s"], mode="markers", name="En alerta", marker=dict(color=RED, size=4))
    for e in sorted(set(ev.loc[ev["v"] == v, "day"] + pd.to_timedelta(ev.loc[ev["v"] == v, "tte"], "D"))):
        fig.add_vline(x=e, line=dict(color=INK2, width=1, dash="dot"))
    st.plotly_chart(style(fig, 320, yaxis_tickformat=".0%"), width="stretch")
    st.caption("Líneas punteadas: eventos de DPF identificados en taller.")

    st.subheader("Datos generados: último día de cada vehículo del holdout")
    last = test.groupby("v").tail(1).sort_values("score_s", ascending=False)
    st.dataframe(last[["v", "day", "gbm30", "gbm60", "gbm90", "score_s", "alerta", "failed"]].rename(columns={
        "v": "Vehículo", "day": "Día", "gbm30": "P(evento ≤ 30 d)", "gbm60": "P(≤ 60 d)", "gbm90": "P(≤ 90 d)",
        "score_s": "Riesgo suavizado", "alerta": "En alerta", "failed": "Tuvo evento"}),
        hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                                         ("P(evento ≤ 30 d)", "P(≤ 60 d)", "P(≤ 90 d)",
                                                          "Riesgo suavizado")})

    st.page_link("app.py", label="Ver el dashboard completo (ensamble con red, supervivencia y autoencoder)",
                 icon=":material/arrow_forward:")


st.navigation([st.Page(live, title="LightGBM en vivo", default=True),
               st.Page("app.py", title="Dashboard completo")]).run()
