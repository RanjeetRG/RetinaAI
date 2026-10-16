"""Interactive charts for the Streamlit app.

Probability chart (FR-4): one series, so every bar has the same colour and there is no legend; bars are at
most ~20 px thick with a rounded data-end and grow from one baseline; the value sits at each bar tip in ink;
gridlines are hairline and recessive; hovering a bar shows its exact value.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import config
from utils.visuals import GRIDLINE, INK, INK_SECONDARY, SERIES_1, SURFACE

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def probability_chart(probabilities: np.ndarray, title: str = "Probability of each stage") -> go.Figure:
    labels = [f"{k}  {name}" for k, name in enumerate(config.STAGE_NAMES)]
    fig = go.Figure(
        go.Bar(
            x=probabilities,
            y=labels,
            orientation="h",
            marker=dict(color=SERIES_1, cornerradius=4),
            text=[f"{p:.0%}" for p in probabilities],
            textposition="outside",
            textfont=dict(color=INK, size=13),
            cliponaxis=False,
            hovertemplate="Stage %{y}<br>Probability %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=260,
        bargap=0.45,  # ~20 px bars at this height
        margin=dict(l=8, r=48, t=44, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=True,
                     zerolinecolor="#c3c2b7", zerolinewidth=1, tickvals=[0, 0.25, 0.5, 0.75, 1])  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    return fig
