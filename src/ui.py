"""Paleta y estilo de gráficos compartidos por los dashboards."""


BLUE, ORANGE, AQUA = "#2a46b8", "#eb6834", "#1baf7a"  # BLUE = acento del pitch
RED, GRAY = "#e34948", "#8a8984"
STATUS = {"Alto": "#d03b3b", "Medio": "#fab219", "Bajo": "#0ca30c"}
GRID, AXIS, INK2 = "#e1e0d9", "#c3c2b7", "#52514e"


def style(fig, h=320, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK2), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode=kw.pop("hovermode", "x unified"),
                      legend=dict(orientation="h", y=1.12), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    return fig

