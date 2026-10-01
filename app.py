"""DPF Health Copilot — dashboard. Ejecutar: streamlit run app.py"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.copilot import (MIN_MINS, WEEKS, completion, min_recipe, recipe_text, regen_windows, simulate, window_text,
                         whatif_target)
from src.evaluate import group_of
from src.features import feature_cols
from src.ui import AQUA, BLUE, GRAY, INK2, ORANGE, RED, STATUS, style

st.set_page_config("DPF Health Copilot", layout="wide")


@st.cache_data
def load():
    p = pd.read_parquet("data/preds.parquet")
    p["day"] = pd.to_datetime(p["day"])
    p["row"] = np.arange(len(p))
    sc = pd.read_parquet("data/scores.parquet")
    p = p.merge(sc, on=["v", "day"], how="left")
    s = pd.read_parquet("data/static.parquet")
    prof = pd.read_parquet("data/profiles.parquet")[["v", "profile"]]
    M = json.load(open("data/metrics.json"))
    q = json.load(open("data/quality.json"))
    return p, s, prof, M, q


@st.cache_data
def load_features():
    f = pd.read_parquet("data/features.parquet")
    f["day"] = pd.to_datetime(f["day"])
    return f


@st.cache_resource
def explainer():
    return shap.TreeExplainer(lgb.Booster(model_file="models/gbm90.txt"))


@st.cache_resource
def whatif_model():
    return lgb.Booster(model_file="models/gbm90_whatif.txt")


@st.cache_data
def whatif_targets():
    """Objetivo de la receta por fila de features: umbral de flota (últimos 30 días) en la escala del GBM what-if."""
    f = load_features()
    return whatif_target(f, whatif_model().predict(f[feature_cols(f)]), REL)


@st.cache_data
def trips():
    return pd.read_parquet("data/trips.parquet", columns=["v", "lts", "mins", "dpf_state0", "dpf_state1"])


@st.cache_data
def regen_completion():
    return completion(trips())


@st.cache_data
def attention():
    return np.load("models/gru_attention.npy")


p, static, prof, M, Q = load()
REL = M["relative_operating_pct"]  # política de alerta: top REL de riesgo de la flota en los últimos 30 días
OP = next(c for c in M["relative_curve"] if c["pct"] == REL)


def status(x, thr, thr_med):
    return "Alto" if x >= thr else ("Medio" if x >= thr_med else "Bajo")


latest = p.sort_values("day").groupby("v").tail(1).merge(static[["v", "country", "ModelSeries", "Engine"]], on="v") \
    .merge(prof, on="v", how="left")
latest["estado"] = [status(*r) for r in latest[["score_s", "thr_rel", "thr_rel_med"]].values]

st.sidebar.title("DPF Health Copilot")
st.sidebar.caption("Predicción temprana de degradación de combustión / DPF a partir de telemetría conectada.")
view = st.sidebar.radio("Vista", ["Flota", "Vehículo", "Modelo y negocio", "Calidad de datos"])
subset = st.sidebar.selectbox("Vehículos", ["Holdout (nunca vistos)", "Todos (OOF)"])
if subset.startswith("Holdout"):
    latest = latest[latest["fold"] == -1]

ADVICE = {
    "Patrón de uso": "Predominan viajes cortos/urbanos: sugerir al cliente un trayecto de 20 min o más esta semana "
                     "(con esa duración termina más del 90 % de las regeneraciones en curso).",
    "Regeneraciones": "Regeneraciones interrumpidas o poco frecuentes: evitar apagar el motor durante la limpieza "
                      "automática; agendar regeneración asistida.",
    "Hollín / DPF": "Carga de hollín en aumento sostenido: programar regeneración forzada en concesionario antes de "
                    "que el DPF llegue al límite.",
    "Térmico / arranques en frío": "Muchos arranques en frío sin alcanzar temperatura de operación: combinar trayectos "
                                   "cortos en uno más largo.",
    "Consumo": "Consumo por encima de su historial: revisar filtro de aire / admisión y presión de neumáticos.",
    "Aceite": "Degradación acelerada del aceite (posible dilución por post-inyección): adelantar el cambio de aceite.",
    "Clima": "Operación en clima frío: mayor tiempo de calentamiento, priorizar trayectos más largos.",
}

# =====================================================================================
if view == "Flota":
    st.title("Estado de la flota")
    c = st.columns(4)
    c[0].metric("Vehículos monitoreados", f"{len(latest)}")
    c[1].metric("En alerta alta", f"{(latest['estado'] == 'Alto').sum()}")
    c[2].metric("Anticipación mediana",
                f"{OP['median_lead_days']:.0f} días · {OP['median_lead_km']:,.0f} km".replace(",", "."),
                f"{OP['median_lead_days'] - (M['ecu_baseline']['median_lead_days'] or 0):+.0f} días vs advertencia ECU")
    c[3].metric("Eventos detectados antes de ocurrir", f"{OP['detection_rate']:.0%}")

    l, r = st.columns([3, 2])
    with l:
        st.subheader("Ranking de riesgo (último día reportado)")
        tb = latest.sort_values("score_s", ascending=False)[
            ["v", "estado", "score_s", "health", "rul", "profile", "country", "ModelSeries", "day"]].copy()
        st.dataframe(tb.rename(columns={"v": "Vehículo", "score_s": "Riesgo 90d", "health": "Health Index",
                                        "rul": "RUL (días)", "profile": "Perfil", "country": "País",
                                        "ModelSeries": "Modelo", "day": "Último dato"}),
                     hide_index=True, height=440,
                     column_config={"Riesgo 90d": st.column_config.ProgressColumn(format="%.2f", min_value=0,
                                                                                  max_value=1),
                                    "Health Index": st.column_config.NumberColumn(format="%.0f"),
                                    "RUL (días)": st.column_config.NumberColumn(format="%.0f")})
    with r:
        st.subheader("Tasa de falla por perfil de conductor")
        pr = pd.DataFrame(M["profiles"]).T.reset_index().rename(columns={"index": "perfil"}).sort_values("failure_rate")
        fig = go.Figure(go.Bar(x=pr["failure_rate"], y=pr["perfil"], orientation="h", marker_color=BLUE,
                               text=[f"{x:.0%} (n={int(n)})" for x, n in zip(pr["failure_rate"], pr["n"])],
                               textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, 260, hovermode="closest", xaxis_tickformat=".0%"), width="stretch")
        st.subheader("Vehículos por estado y país")
        ct = latest.groupby(["country", "estado"]).size().unstack(fill_value=0)
        fig = go.Figure([go.Bar(name=s, x=ct.index, y=ct.get(s, 0), marker_color=STATUS[s],
                                marker_line=dict(color="white", width=2)) for s in ["Alto", "Medio", "Bajo"]])
        st.plotly_chart(style(fig, 260, barmode="stack"), width="stretch")

# =====================================================================================
elif view == "Vehículo":
    f = load_features()
    cols = feature_cols(f)
    opts = latest.sort_values(["failed", "score_s"], ascending=False)["v"].tolist()
    failed = static.set_index("v")["failed"]
    v = st.selectbox("Vehículo", opts, format_func=lambda x: f"{x} — {'con evento' if failed[x] else 'sano'}")
    pv = p[p["v"] == v].sort_values("day")
    fv = f[f["v"] == v].sort_values("day")
    events = static.set_index("v").loc[v, "events"]
    day = st.select_slider("Fecha de análisis", options=list(pv["day"].dt.date), value=pv["day"].dt.date.iloc[-1])
    cur = pv[pv["day"].dt.date == day].iloc[0]
    fx = fv[fv["day"].dt.date == day]

    c = st.columns(5)
    s_ = status(cur["score_s"], cur["thr_rel"], cur["thr_rel_med"])
    c[0].metric("Estado", s_)
    c[1].metric("Riesgo evento 90 días", f"{cur['stack90']:.0%}")
    c[2].metric("Riesgo 30 días", f"{cur['stack30']:.0%}")
    c[3].metric("Health Index", f"{cur['health']:.0f}/100")
    c[4].metric("RUL estimado (supervivencia)", f"{cur['rul']:.0f} días")

    fig = go.Figure()
    fig.add_scatter(x=pv["day"], y=pv["score_s"], name="Riesgo 90d (ensamble, media 7d)",
                    line=dict(color=BLUE, width=2))
    fig.add_scatter(x=pv["day"], y=pv["gru90"], name="GRU", line=dict(color=AQUA, width=1), opacity=0.6)
    ecu = pv[pv["ecu_warning"] > 0]
    fig.add_scatter(x=ecu["day"], y=np.full(len(ecu), 0.02), mode="markers", name="Advertencia ECU (DPF sobre límite)",
                    marker=dict(color=ORANGE, size=8, symbol="triangle-up"))
    fig.add_scatter(x=pv["day"], y=pv["thr_rel"], name=f"umbral de flota (top {REL:.0%} últimos 30 días)",
                    line=dict(color=GRAY, dash="dash", width=1))
    for e in events:
        fig.add_vline(x=pd.Timestamp(e), line=dict(color=RED, width=2))
        fig.add_annotation(x=pd.Timestamp(e), y=1, yref="paper", text="evento identificado", showarrow=False,
                           font=dict(color=RED), xanchor="left")
    fig.add_vline(x=pd.Timestamp(day), line=dict(color=INK2, width=1, dash="dot"))
    st.plotly_chart(style(fig, 340, title="Evolución del riesgo", yaxis_range=[0, 1]), width="stretch")

    # small multiples de señales físicas (sin doble eje)
    sig = [("w7_acc_mean", "Acumulación de hollín (media 7d, %)"),
           ("w30_km_per_regen", "Km entre regeneraciones (30d)"), ("w30_sh_short5", "Viajes < 5 km (30d)"),
           ("w30_fuel_per100", "Consumo (% tanque /100 km, 30d)")]
    cc = st.columns(4)
    for col, (s, t) in zip(cc, sig):
        fig = go.Figure(go.Scatter(x=fv["day"], y=fv[s], line=dict(color=BLUE, width=2), name=t))
        for e in events:
            fig.add_vline(x=pd.Timestamp(e), line=dict(color=RED, width=1))
        col.plotly_chart(style(fig, 200, title=dict(text=t, font=dict(size=12)), showlegend=False), width="stretch")

    l, m_, r = st.columns([2, 1.2, 1.3])
    ex = explainer()
    X = fx[cols]
    sv = ex.shap_values(X)
    sv = (sv[1] if isinstance(sv, list) else sv)[0]
    contrib = pd.DataFrame({"feature": cols, "shap": sv, "value": X.iloc[0].values}).assign(a=lambda d: d.shap.abs())
    top = contrib.nlargest(12, "a").sort_values("shap")
    with l:
        st.subheader("¿Por qué este riesgo? (SHAP)")
        fig = go.Figure(go.Bar(x=top["shap"], y=top["feature"], orientation="h",
                               marker_color=[RED if s > 0 else BLUE for s in top["shap"]],
                               customdata=top["value"],
                               hovertemplate="%{y}<br>valor=%{customdata:.3g}<br>SHAP=%{x:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 380, hovermode="closest", xaxis_title="← baja el riesgo | sube el riesgo →"),
                        width="stretch")
    with m_:
        st.subheader("Supervivencia (RSF)")
        fig = go.Figure(go.Bar(x=["30d", "60d", "90d", "180d"], y=[cur[f"rsf{t}"] for t in (30, 60, 90, 180)],
                               marker_color=BLUE, text=[f"{cur[f'rsf{t}']:.0%}" for t in (30, 60, 90, 180)],
                               textposition="outside", hovertemplate="P(evento ≤ %{x}) = %{y:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, 380, hovermode="closest", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                              yaxis_title="P(evento antes de t)"), width="stretch")
    with r:
        st.subheader("Atención de la GRU")
        att = attention()[int(cur["row"])].astype(float)
        days = pd.date_range(end=pd.Timestamp(day), periods=len(att))
        fig = go.Figure(go.Bar(x=days, y=att, marker_color=AQUA, hovertemplate="%{x|%d-%b}: %{y:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 380, hovermode="closest", yaxis_title="peso de atención",
                              title=dict(text=f"Días de los últimos {len(att)} que más pesaron", font=dict(size=12))),
                        width="stretch")

    st.subheader("Recomendación prescriptiva")
    drivers = contrib[contrib["shap"] > 0].assign(g=lambda d: d.feature.map(group_of)).groupby("g")["shap"].sum() \
        .sort_values(ascending=False)
    for g in [g for g in drivers.index if g in ADVICE][:3]:
        st.markdown(f"- **{g}** → {ADVICE[g]}")
    if drivers.empty:
        st.markdown("- Uso saludable: sin acciones necesarias.")

    st.subheader("Receta mínima: el cambio más fácil que saca al vehículo de alerta")
    wm = whatif_model()
    rec = min_recipe(wm, X, whatif_targets()[int(fx.index[0])])
    st.markdown(recipe_text(rec))
    st.caption("Busca entre todas las combinaciones del simulador (trayectos de ruta, menos viajes cortos, no cortar "
               "la limpieza) la de menor esfuerzo que lleva el riesgo por debajo del umbral de la flota "
               f"(top {REL:.0%} de los últimos 30 días), con el GBM de restricciones físicas.")

    st.subheader("Ventana de regeneración: cuándo le conviene hacerlo")
    P, N = regen_windows(trips().loc[lambda t: t["v"] == v], day)
    st.markdown(window_text(P, N))
    l2, r2 = st.columns([3, 2])
    with l2:
        fig = go.Figure(go.Heatmap(z=P.values, x=P.columns, y=P.index, zmin=0, zmax=1,
                                   colorscale=[[0, "#f4f3ee"], [1, AQUA]], colorbar=dict(tickformat=".0%"),
                                   hovertemplate="%{y} %{x}: trayecto apto en %{z:.0%} de las semanas<extra></extra>"))
        st.plotly_chart(style(fig, 300, hovermode="closest", yaxis_autorange="reversed",
                              title=dict(text=f"Semanas con un trayecto de {MIN_MINS}+ min (últimas {WEEKS})",
                                         font=dict(size=12))), width="stretch")
    with r2:
        cp = regen_completion()
        fig = go.Figure(go.Bar(x=[f"{int(i.left)}–{int(i.right)}" if i.right < 1e3 else f"{int(i.left)}+"
                                  for i in cp.index], y=cp["mean"], marker_color=BLUE, customdata=cp["size"],
                               hovertemplate="%{x} min: %{y:.0%} (n=%{customdata})<extra></extra>"))
        st.plotly_chart(style(fig, 300, hovermode="closest", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                              xaxis_title="duración del viaje (min)",
                              title=dict(text="Regeneraciones en curso que terminan dentro del viaje (toda la flota)",
                                         font=dict(size=12))), width="stretch")

    st.subheader("Simulador what-if: ¿qué pasa si cambia el hábito de manejo?")
    st.caption("Usa un GBM con restricciones monótonas físicas (más viajes cortos / regeneraciones interrumpidas nunca "
               "bajan el riesgo; viajes más largos y rápidos nunca lo suben), para que la simulación sea coherente.")
    w = st.columns(3)
    long_trips = w[0].slider("Viajes de ruta (≥ 40 km) extra por semana", 0, 5, 1)
    short_cut = w[1].slider("Reducir viajes cortos (< 5 km) en", 0, 100, 0, format="%d%%")
    warm = w[2].checkbox("Evitar apagar el motor durante la regeneración", value=False)
    Xs = simulate(X, long_trips, short_cut, warm)
    p0, p1 = wm.predict(X)[0], wm.predict(Xs)[0]
    c = st.columns(3)
    c[0].metric("Riesgo actual (modelo what-if)", f"{p0:.0%}")
    c[1].metric("Riesgo con el nuevo hábito", f"{p1:.0%}", f"{(p1 - p0) * 100:+.0f} pp", delta_color="inverse")
    c[2].metric("Reducción relativa", f"{(1 - p1 / p0) if p0 > 0 else 0:.0%}")

# =====================================================================================
elif view == "Modelo y negocio":
    st.title("Desempeño del modelo (holdout de vehículos nunca vistos)")
    d = pd.DataFrame(M["discrimination"])
    st.subheader("Discriminación por modelo y horizonte")
    piv = d.pivot(index="model", columns="H", values=["auc", "ap", "auc_within_failed"]).round(3)
    piv.columns = [f"{a.upper()} {h}d" for a, h in piv.columns]
    st.dataframe(piv, width="stretch")
    st.caption(f"AUC within-failed: discrimina *cuándo* se acerca el evento dentro de vehículos que fallan "
               f"(inmune a diferencias de cohorte). C-index supervivencia (RSF): {M['rsf_c_index']:.3f}")
    ci = d[d["model"] == "stack"][["H", "auc", "auc_ci95", "ap", "ap_ci95", "oof_auc"]].copy()
    ci["AUC holdout (IC95 por vehículo)"] = [f"{a:.3f} [{c[0]:.3f}–{c[1]:.3f}]"
                                             for a, c in zip(ci["auc"], ci["auc_ci95"])]
    ci["AP holdout (IC95)"] = [f"{a:.3f} [{c[0]:.3f}–{c[1]:.3f}]" for a, c in zip(ci["ap"], ci["ap_ci95"])]
    ci["AUC fuera de fold (train)"] = ci["oof_auc"].round(3)
    st.dataframe(ci.drop(columns=["auc", "auc_ci95", "ap", "ap_ci95", "oof_auc"])
                 .rename(columns={"H": "Horizonte (días)"}), hide_index=True, width="stretch")

    try:
        TV = json.load(open("data/temporal.json"))
        st.subheader(f"Validación temporal: sistema desplegado el {TV['T']}")
        st.caption("Entrena solo con etiquetas conocidas antes de esa fecha y evalúa después (LightGBM).")
        rows = []
        for k, name in [("vehiculos_nuevos", "Vehículos nuevos (holdout)"),
                        ("misma_flota", "Misma flota, en el futuro")]:
            for h in (30, 60, 90):
                r = TV[k][f"H{h}"]
                rows.append({"Escenario": name, "Horizonte": h,
                             "AUC (IC95)": f"{r['auc']:.3f} [{r['auc_ci95'][0]:.3f}–{r['auc_ci95'][1]:.3f}]",
                             "AP (base)": f"{r['ap']:.3f} ({r['base_rate']:.3f})", "Vehículos": r["n_vehicles"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    except FileNotFoundError:
        st.info("Ejecutar `python -m src.temporal` para ver la validación temporal.")

    l, r = st.columns(2)
    with l:
        st.subheader("Detección vs falsas alarmas")
        st.caption(f"Política de alerta: top {REL:.0%} de riesgo de la flota en los últimos 30 días (umbral relativo), "
                   "elegida fuera de fold con la misma tasa de falsas alarmas que la ECU, sin mirar el holdout.")
        fig = go.Figure()
        for key, name, color in [("leadtime_curve", "Umbral fijo", BLUE),
                                 ("relative_curve", "Umbral relativo a la flota", AQUA)]:
            lc = pd.DataFrame(M[key])
            fig.add_scatter(x=lc["false_alarm_episodes_per_vehicle_year"], y=lc["detection_rate"],
                            mode="lines+markers", name=name, line=dict(color=color, width=2), marker=dict(size=8),
                            customdata=lc[["median_lead_days", "median_lead_km"]],
                            hovertemplate="falsas alarmas/vehículo-año=%{x:.2f}<br>detección=%{y:.0%}"
                                          "<br>anticipación mediana=%{customdata[0]:.0f} días"
                                          " · %{customdata[1]:,.0f} km"
                                          "<extra></extra>")
        e = M["ecu_baseline"]
        fig.add_scatter(x=[e["false_alarm_episodes_per_vehicle_year"]], y=[e["detection_rate"]], mode="markers+text",
                        name="Advertencia ECU actual", marker=dict(color=ORANGE, size=11, symbol="diamond"),
                        text=["ECU"], textposition="bottom right",
                        hovertemplate="ECU: falsas alarmas=%{x:.2f}<br>detección=%{y:.0%}<extra></extra>")
        fig.add_scatter(x=[OP["false_alarm_episodes_per_vehicle_year"]], y=[OP["detection_rate"]], mode="markers+text",
                        name="Punto de operación", marker=dict(color=AQUA, size=14, symbol="circle-open", line_width=3),
                        text=[f"{OP['detection_rate']:.0%}"], textposition="top left", hoverinfo="skip")
        st.plotly_chart(style(fig, 380, hovermode="closest", xaxis_title="episodios de falsa alarma por vehículo-año",
                              yaxis_title="eventos detectados antes de ocurrir", yaxis_tickformat=".0%"),
                        width="stretch")
    with r:
        st.subheader("Calibración (riesgo 90 días)")
        cal = pd.DataFrame(M["calibration"])
        fig = go.Figure()
        fig.add_scatter(x=[0, cal["pred"].max()], y=[0, cal["pred"].max()], name="ideal",
                        line=dict(color=GRAY, dash="dash", width=1))
        fig.add_scatter(x=cal["pred"], y=cal["obs"], name="ensamble", mode="lines+markers",
                        line=dict(color=BLUE, width=2))
        st.plotly_chart(style(fig, 380, hovermode="closest", xaxis_title="probabilidad predicha",
                              yaxis_title="frecuencia observada"), width="stretch")

    l, r = st.columns(2)
    sg = pd.read_parquet("data/shap_global.parquet")
    with l:
        st.subheader("Qué explica el riesgo (SHAP por hipótesis física)")
        g = pd.Series(M["shap_by_group"]).sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 360, hovermode="closest", xaxis_title="Σ |SHAP| medio"), width="stretch")
    with r:
        st.subheader("Top 15 features")
        t = sg.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["mean_abs_shap"], y=t["feature"], orientation="h", marker_color=BLUE,
                               customdata=t["group"], hovertemplate="%{y} (%{customdata}): %{x:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 360, hovermode="closest"), width="stretch")

    st.subheader("Impacto económico estimado")
    c = st.columns(3)
    fleet = c[0].number_input("Vehículos diésel en la red", 1000, 1_000_000, 50_000, step=1000)
    rate = c[1].number_input("Eventos de degradación por vehículo-año", 0.0, 0.5, 0.05, format="%.3f")
    success = c[2].slider("Eventos evitados cuando se alerta a tiempo", 0, 100, 60, format="%d%%") / 100
    c = st.columns(3)
    repair = c[0].number_input("Costo correctivo por evento (USD: garantía + DPF + grúa)", 0, 10000, 1800, step=100)
    notify = c[1].number_input("Costo por alerta nivel 1 (push al cliente + seguimiento, USD)", 0, 200, 3)
    prev = c[2].number_input("Costo acción preventiva nivel 2 (regeneración asistida, USD)", 0, 2000, 120, step=10)
    ev = fleet * rate
    det = ev * OP["detection_rate"]
    alerts = det + fleet * OP["false_alarm_episodes_per_vehicle_year"]
    # nivel 1: toda alerta = notificación con recomendación de manejo; nivel 2: solo eventos reales que persisten
    saving = det * success * repair - alerts * notify - det * prev
    c = st.columns(4)
    c[0].metric("Eventos anticipados / año", f"{det:,.0f}")
    c[1].metric("Eventos evitados / año", f"{det * success:,.0f}")
    c[2].metric("Alertas nivel 1 / año", f"{alerts:,.0f}")
    c[3].metric("Ahorro neto estimado / año", f"USD {saving:,.0f}")
    st.caption("Supuestos editables; tasas de detección y falsa alarma medidas en holdout con el umbral del punto de "
               "operación (días-sanos en alerta).")

# =====================================================================================
else:
    st.title("Calidad y trazabilidad del pipeline")
    st.markdown("""
**Decisiones de datos**
- **Etiquetas**: se usan los archivos *v2* de fallados. En v1, `IdentificationDate == daysUntilSale` en el 76% de los
  casos (fecha de venta, no de falla).
- **Anclaje temporal**: la fecha de producción se reconstruye como `D0 + ProductionDay`; el primer viaje (en planta)
  coincide con producción con dispersión p5–p95 < 1 día, lo que valida el anclaje y permite ubicar el evento en el
  calendario.
- **Conflictos de etiqueta**: sanos que aparecen en alguna lista de fallados → excluidos.
- **Eventos múltiples**: un vehículo puede tener varios eventos; tras cada service se excluyen 14 días (transición) y el
  reloj se reinicia.
- **Censura**: un día sano solo es negativo si se observan los H días siguientes.
- **Anti-leakage**: features solo con pasado (test automático de historia truncada); edad, odómetro y fecha nunca son
  features (los fallados son una cohorte más antigua); split por vehículo con holdout del 20%.
- **Corte de telemetría detectado**: desde el {} no llega ningún evento de regeneración (`Regenerations`) en toda la
  flota, y ya venía perdiendo eventos desde marzo. Usado tal cual, el riesgo de los vehículos sanos subía
  artificialmente de 4% a 41%.
- **Regeneraciones reconstruidas**: se detectan como caídas de hollín (`Acumulation`) ≥ 20 puntos entre lecturas
  consecutivas, señal que nunca dejó de llegar. Validado contra la bandera antes del corte: recall 0.91, precisión 0.81,
  correlación vehículo-mes 0.88, y tasa estable (≈2.5–3 por 1000 km) antes y después del corte. La distancia entre
  regeneraciones sale del odómetro. Esto recupera todo el período sin cortes.
- **Sentinelas**: temperaturas −73/−128 °C, −60 °C de motor y valores fuera de rango físico → nulos.
""".format(Q["regen_flag_outage_from"]))
    rows = [(k, v) for k, v in Q.items() if not isinstance(v, dict)]
    st.dataframe(pd.DataFrame(rows, columns=["control", "valor"]).astype(str), hide_index=True, width="stretch")
    st.subheader("Tasa de nulos por variable de viaje (post-limpieza)")
    nr = pd.Series(Q["trip_null_rate"]).sort_values()
    fig = go.Figure(go.Bar(x=nr.values, y=nr.index, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    st.plotly_chart(style(fig, 260, hovermode="closest", xaxis_tickformat=".1%"), width="stretch")
