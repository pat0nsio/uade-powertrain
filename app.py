"""Powertrain Health Copilot — dashboard. Ejecutar: streamlit run app.py"""
import json

import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.copilot import min_recipe, recipe_text, regen_windows, simulate, window_text, whatif_target
from src.evaluate import group_of
from src.features import feature_cols
from src.ui import BLUE, RED, detection_chart, risk_chart, style, weight_chart

st.set_page_config("Powertrain Health Copilot", layout="wide")


@st.cache_data
def load():
    p = pd.read_parquet("data/preds.parquet", columns=["v", "day", "fold", "failed", "stack90", "health"])
    p["day"] = pd.to_datetime(p["day"])
    p = p.merge(pd.read_parquet("data/scores.parquet"), on=["v", "day"], how="left")
    n_days = len(p)  # vehículo-días del estudio
    p = p[p["fold"] == -1]  # solo los 198 vehículos que el modelo nunca vio
    return p, pd.read_parquet("data/static.parquet"), json.load(open("data/metrics.json")), \
        json.load(open("data/quality.json")), n_days


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


p, static, M, Q, N_DAYS = load()
REL = M["relative_operating_pct"]  # política de alerta: top REL de riesgo de la flota en los últimos 30 días
DEMO = "VEH_0543"  # la unidad de Ibagué que sigue el pitch: alerta 129 días antes del evento
OP, ECU = next(c for c in M["relative_curve"] if c["pct"] == REL), M["ecu_baseline"]
pct = lambda x: f"{x:.0%}"
num = lambda x: f"{x:,.0f}".replace(",", ".")
dec = lambda x: f"{x:.2f}".replace(".", ",")


def status(x, thr, thr_med):
    return "Alto" if x >= thr else ("Medio" if x >= thr_med else "Bajo")


latest = p.sort_values("day").groupby("v").tail(1).merge(static[["v", "country"]], on="v")
latest["estado"] = [status(*r) for r in latest[["score_s", "thr_rel", "thr_rel_med"]].values]

st.sidebar.title("Powertrain Health Copilot")
view = st.sidebar.radio("Vista", ["Flota", "Vehículo", "Resultados", "Datos"])
st.sidebar.caption(f"{latest['v'].nunique()} vehículos que el modelo nunca vio durante el entrenamiento.")

# =====================================================================================
if view == "Flota":
    st.title("Estado de la flota")
    c = st.columns(3)
    c[0].metric("Vehículos monitoreados", len(latest))
    c[1].metric("En alerta", int((latest["estado"] == "Alto").sum()))
    c[2].metric("Eventos anticipados", pct(OP["detection_rate"]), f"la ECU hoy: {pct(ECU['detection_rate'])}",
                delta_color="off", delta_arrow="off")

    st.subheader("A quién llamar primero")
    tb = latest.sort_values("score_s", ascending=False)[["v", "estado", "score_s", "country", "day"]] \
        .assign(score_s=lambda d: d["score_s"] * 100)
    st.dataframe(tb.rename(columns={"v": "Vehículo", "estado": "Estado", "score_s": "Riesgo a 90 días",
                                    "country": "País", "day": "Último dato"}),
                 hide_index=True, height=520, width="stretch",
                 column_config={"Riesgo a 90 días": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                                    max_value=100),
                                "Último dato": st.column_config.DateColumn(format="DD/MM/YYYY")})
    st.caption(f"Estado alto = el vehículo está en el {REL:.0%} de mayor riesgo de la flota en los últimos 30 días.")

# =====================================================================================
elif view == "Vehículo":
    f = load_features()
    cols = feature_cols(f)
    failed = static.set_index("v")["failed"]
    opts = latest.sort_values(["failed", "score_s"], ascending=False)["v"].tolist()
    v = st.selectbox("Vehículo", opts, index=opts.index(DEMO) if DEMO in opts else 0, format_func=lambda x: f"{x} · {'tuvo un evento' if failed[x] else 'sano'}")
    pv = p[p["v"] == v].sort_values("day")
    fv = f[f["v"] == v].sort_values("day")
    events = [pd.Timestamp(e) for e in static.set_index("v").loc[v, "events"]]

    # por defecto se analiza el día de la primera alerta antes del primer evento (lo que habría visto el taller)
    lead, default = None, pv["day"].iloc[-1]
    if events:
        pre = pv[(pv["day"] < events[0]) & (pv["day"] >= events[0] - pd.Timedelta(days=180))]
        on = pre[pre["score_s"] >= pre["thr_rel"]]
        if len(on):
            default, lead = on["day"].iloc[0], (events[0] - on["day"].iloc[0]).days
    day = pd.Timestamp(st.select_slider("Día analizado", options=list(pv["day"].dt.date), value=default.date()))
    cur = pv[pv["day"] == day].iloc[0]
    X = fv.loc[fv["day"] == day, cols]

    c = st.columns(3)
    c[0].metric("Estado ese día", status(cur["score_s"], cur["thr_rel"], cur["thr_rel_med"]))
    c[1].metric("Riesgo de evento en 90 días", pct(cur["stack90"]))
    c[2].metric("Primera alerta antes del evento", f"{'más de 180' if lead >= 180 else lead} días" if lead
                else "sin alerta" if events else "no tuvo evento")

    st.plotly_chart(risk_chart(pv["day"], pv["score_s"], pv["thr_rel"], pv.loc[pv["ecu_warning"] > 0, "day"], events,
                               day), width="stretch")

    l, r = st.columns(2)
    with l:
        st.subheader("Qué hábitos empujan el riesgo")
        sv = explainer().shap_values(X)
        sv = (sv[1] if isinstance(sv, list) else sv)[0]
        g = pd.Series(sv, index=cols).groupby(group_of).sum().drop(["Vehículo / mercado", "Otros"], errors="ignore")
        g = g[g.abs().nlargest(5).index].sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", hoverinfo="skip",
                               marker_color=[RED if x > 0 else BLUE for x in g.values]))
        st.plotly_chart(style(fig, 280, xaxis_title="← baja el riesgo · sube el riesgo →", xaxis_showticklabels=False),
                        width="stretch")
        st.caption("Sin contar el país y el modelo del vehículo, que no dependen del conductor.")
    with r:
        st.subheader("Qué hacer")
        wm = whatif_model()
        st.markdown(recipe_text(min_recipe(wm, X, whatif_targets()[int(X.index[0])])))
        st.markdown(window_text(*regen_windows(trips().loc[lambda t: t["v"] == v], day)))

    st.subheader("Probar un cambio de hábito")
    w = st.columns(3)
    long_trips = w[0].slider("Viajes de ruta extra por semana", 0, 5, 1)
    short_cut = w[1].slider("Menos viajes de menos de 5 km", 0, 100, 0, step=25, format="%d%%")
    warm = w[2].checkbox("No apagar el motor durante la limpieza del filtro")
    p0, p1 = wm.predict(X)[0], wm.predict(simulate(X, long_trips, short_cut, warm))[0]
    st.metric("Cuánto baja el riesgo con ese cambio", pct(1 - p1 / p0) if p0 > 0 else "0%")

# =====================================================================================
elif view == "Resultados":
    st.title("Cuánto mejor que la advertencia actual")
    c = st.columns(3)
    c[0].metric("Eventos anticipados", pct(OP["detection_rate"]), f"ECU: {pct(ECU['detection_rate'])}",
                delta_color="off", delta_arrow="off")
    c[1].metric("Anticipación mediana", f"{OP['median_lead_days']:.0f} días · {num(OP['median_lead_km'])} km")
    c[2].metric("Falsas alarmas por vehículo y año", dec(OP["false_alarm_episodes_per_vehicle_year"]),
                f"ECU: {dec(ECU['false_alarm_episodes_per_vehicle_year'])}", delta_color="off", delta_arrow="off")

    l, r = st.columns(2)
    with l:
        st.subheader("Más eventos detectados con las mismas falsas alarmas")
        st.plotly_chart(detection_chart(M["relative_curve"], ECU, OP), width="stretch")
    with r:
        st.subheader("Qué pesa en el riesgo")
        st.plotly_chart(weight_chart(M["shap_by_group"]), width="stretch")

    st.subheader("Ahorro estimado por año")
    c = st.columns(3)
    fleet = c[0].number_input("Vehículos diésel", 1000, 1_000_000, 50_000, step=5000)
    repair = c[1].number_input("Costo de cada evento (USD)", 0, 10000, 1800, step=100)
    success = c[2].slider("Eventos evitados cuando la alerta llega a tiempo", 0, 100, 60, format="%d%%") / 100

    def saving(r, rate=0.05, notify=3, prev=200):  # supuestos fijos del informe (tabla 14)
        det = fleet * rate * r["detection_rate"]
        return det * success * repair - (det + fleet * r["false_alarm_episodes_per_vehicle_year"]) * notify - det * prev

    c = st.columns(2)
    c[0].metric("Ahorro neto con Powertrain Health Copilot", f"USD {num(saving(OP))}")
    c[1].metric("Ahorro adicional frente a la ECU", f"USD {num(saving(OP) - saving(ECU))}")
    st.caption("Supuestos: 0,05 eventos por vehículo y año, USD 3 por aviso al cliente y USD 200 por acción preventiva.")

# =====================================================================================
else:
    st.title("De dónde salen los datos")
    c = st.columns(4)
    c[0].metric("Viajes válidos", num(Q["trips_clean"]), f"−{num(Q['trips_raw'] - Q['trips_clean'])} descartados",
                delta_color="off", delta_arrow="off")
    c[1].metric("Mensajes de la ECU", num(Q["dynamic_rows"]))
    c[2].metric("Limpiezas reconstruidas", num(Q["regen_reconstructed_episodes"]))
    c[3].metric("Vehículo-días", num(N_DAYS))
    st.markdown(f"""
- **Viajes descartados:** {num(Q['trips_duplicates_dropped'])} duplicados y {num(Q['trips_dropped_invalid_total'])}
  imposibles (odómetro que retrocede, más de 200 km/h o más de 24 h).
- **Etiquetas:** solo la lista v2 de fallados; en la v1 la fecha de falla era la fecha de venta.
- **Corte de señal:** desde el {pd.Timestamp(Q['regen_flag_outage_from']):%d/%m/%Y} no llegan los eventos de regeneración. Las limpiezas se
  reconstruyen desde la carga de hollín, que nunca se cortó.
- **Sin trampas:** las variables solo miran el pasado, y la edad, el odómetro y la fecha no se usan.
""")
