"""Paleta, estilo y gráficos compartidos por los dashboards."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA = "#2a46b8", "#eb6834", "#1baf7a"  # BLUE = acento del pitch
RED, GRAY = "#e34948", "#8a8984"
GRID, AXIS, INK2 = "#e1e0d9", "#c3c2b7", "#52514e"


def style(fig, h=320, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK2), separators=",.", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode=kw.pop("hovermode", "x unified"),
                      legend=dict(orientation="h", y=1.12), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    return fig


def detection_chart(curve, ecu, op):
    """Detección vs falsas alarmas de la política relativa, con la ECU y el punto de operación."""
    lc, fa = pd.DataFrame(curve), "false_alarm_episodes_per_vehicle_year"
    fig = go.Figure()
    fig.add_scatter(x=lc[fa], y=lc["detection_rate"], mode="lines+markers", line=dict(color=AQUA, width=2.5),
                    marker=dict(size=7))
    fig.add_scatter(x=[ecu[fa]], y=[ecu["detection_rate"]], mode="markers+text", textposition="bottom right",
                    text=[f"ECU {ecu['detection_rate']:.0%}"], marker=dict(color=ORANGE, size=13, symbol="diamond"))
    fig.add_scatter(x=[op[fa]], y=[op["detection_rate"]], mode="markers+text", textposition="top left",
                    text=[f"{op['detection_rate']:.0%}"],
                    marker=dict(color=AQUA, size=16, symbol="circle-open", line_width=3))
    return style(fig, 360, hovermode=False, showlegend=False, yaxis_tickformat=".0%",
                 xaxis_title="falsas alarmas por vehículo sano y año", yaxis_title="eventos detectados antes")


def weight_chart(by_group):
    """Peso de cada grupo de variables, en % del |SHAP| total."""
    g = pd.Series(by_group).drop("Otros", errors="ignore")
    g = (g / g.sum()).sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=BLUE, text=[f"{x:.0%}" for x in g],
                           textposition="outside", hoverinfo="skip"))
    return style(fig, 360, xaxis_visible=False, xaxis_range=[0, g.max() * 1.2])


def risk_chart(days, risk, thr, ecu_days, events, mark=None):
    """Riesgo de un vehículo en el tiempo, con el umbral de la flota, avisos de la ECU y eventos."""
    fig = go.Figure()
    fig.add_scatter(x=days, y=thr, name="umbral de alerta", line=dict(color=GRAY, dash="dash", width=1))
    fig.add_scatter(x=days, y=risk, name="riesgo del vehículo", line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=ecu_days, y=np.full(len(ecu_days), 0.01), mode="markers", name="advertencia de la ECU",
                    marker=dict(color=ORANGE, size=9, symbol="triangle-up"))
    for e in events:
        fig.add_vline(x=e, line=dict(color=RED, width=2))
        fig.add_annotation(x=e, y=1, yref="paper", text="evento", showarrow=False, font=dict(color=RED),
                           xanchor="left")
    if mark is not None:
        fig.add_vline(x=mark, line=dict(color=INK2, width=1, dash="dot"))
    return style(fig, 340, yaxis_tickformat=".0%", yaxis_rangemode="tozero")
