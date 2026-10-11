"""Interactive Plotly visualizations for island-biogeography runs.

Interactive, animated counterparts to the reference ``plots/plot_islands.py``
(spatial map, species-richness timeseries, per-island abundance dynamics), plus
the study-specific scatter/network charts. Theming follows the validated
data-viz palette (light surface ``#fcfcfb``, the 8-slot categorical order, and
the blue sequential ramp); identity is never carried by color alone (legends +
direct hover labels are always present).

All functions return a ``plotly.graph_objects.Figure``; :func:`write_html`
renders it with ``include_plotlyjs="cdn"`` so the emitted CDN script matches the
installed ``plotly.py`` major version automatically.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

# --- validated data-viz palette (light mode) ------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# blue sequential ramp (100 -> 700), for magnitude encodings
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
]


def _plotly_colorscale(steps):
    n = len(steps)
    return [[i / (n - 1), c] for i, c in enumerate(steps)]


def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=18)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=INK_SECONDARY, size=13,
        ),
        margin=dict(l=70, r=30, t=60, b=60),
        legend=dict(font=dict(color=INK_SECONDARY)),
        colorway=CATEGORICAL,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY))
    return fig


def write_html(fig: go.Figure, path, title: str = "") -> None:
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )


# --------------------------------------------------------------------------
# helpers to reduce an emitter row-history into convenient arrays
# --------------------------------------------------------------------------
def richness_of(abundance: dict[str, float]) -> int:
    return sum(1 for c in abundance.values() if c > 0)


def history_times(rows: list[dict]) -> np.ndarray:
    return np.array([r.get("time", i) for i, r in enumerate(rows)], dtype=float)


# --------------------------------------------------------------------------
# 1. Animated spatial island map
# --------------------------------------------------------------------------
def spatial_map_animation(rows, island_meta, *, title="Island archipelago over time",
                          color_by="richness"):
    """Animated spatial map: islands positioned by x/y, sized by area, colored by
    per-island species richness (or total abundance), stepping through time.

    ``island_meta`` maps island name -> {"x","y","area"}.
    """
    names = list(island_meta.keys())
    xs = [island_meta[n]["x"] for n in names]
    ys = [island_meta[n]["y"] for n in names]
    areas = [island_meta[n]["area"] for n in names]
    # marker diameter proportional to island radius (reference radius = sqrt(area/pi))
    sizes = [max(8.0, 2.6 * math.sqrt(a / math.pi) + 8.0) for a in areas]
    times = history_times(rows)

    def metric(ab):
        if color_by == "richness":
            return richness_of(ab)
        return sum(ab.values())

    # Subsample to a readable number of animation frames: 150 slider steps
    # render as an unreadable tick ruler, so step through ~24 evenly-spaced
    # generations (first and last always included).
    n_rows = len(rows)
    max_frames = 24
    if n_rows > max_frames:
        frame_idx = sorted(set(int(round(k)) for k in np.linspace(0, n_rows - 1, max_frames)))
    else:
        frame_idx = list(range(n_rows))

    values_over_time = [[metric(r["islands"][n]) for n in names] for r in rows]
    vmax = max((max(v) for v in values_over_time if v), default=1) or 1
    cbar_title = "Species<br>richness" if color_by == "richness" else "Total<br>abundance"

    def frame_trace(vals):
        return go.Scatter(
            x=xs, y=ys, mode="markers+text", text=names,
            textposition="top center", textfont=dict(color=INK_SECONDARY, size=11),
            marker=dict(
                size=sizes, color=vals, cmin=0, cmax=vmax,
                colorscale=_plotly_colorscale(SEQUENTIAL_BLUE),
                line=dict(color=SURFACE, width=2),
                colorbar=dict(title=cbar_title, tickfont=dict(color=MUTED)),
            ),
            customdata=np.column_stack([areas, vals]),
            hovertemplate="<b>%{text}</b><br>area=%{customdata[0]:.0f}<br>" +
                          (("richness" if color_by == "richness" else "abundance") +
                           "=%{customdata[1]:.0f}") + "<extra></extra>",
        )

    fig = go.Figure(
        data=[frame_trace(values_over_time[frame_idx[0]])],
        frames=[
            go.Frame(data=[frame_trace(values_over_time[k])], name=f"{times[k]:.0f}")
            for k in frame_idx
        ],
    )
    # Square data window centered on the archipelago, so equal-aspect geometry
    # (scaleanchor) is satisfied by shrinking the plot DOMAIN to a square
    # (constrain="domain") rather than ballooning an axis range — the latter
    # collapsed every island into the centre on a wide, short container.
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0
    r = max(half * 1.22, 30.0)
    fig.update_xaxes(range=[cx - r, cx + r], title="x position", constrain="domain")
    fig.update_yaxes(range=[cy - r, cy + r], title="y position",
                     scaleanchor="x", scaleratio=1, constrain="domain")

    fig.update_layout(
        height=560,
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=1.12, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 350, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 150}}]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.1, len=0.85, y=0, pad=dict(t=10, b=10),
            currentvalue=dict(prefix="generation = ", font=dict(color=INK)),
            tickcolor=GRID, ticklen=4, minorticklen=0, font=dict(color=MUTED, size=10),
            steps=[dict(method="animate", label=f"{times[k]:.0f}",
                        args=[[f"{times[k]:.0f}"],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate"}])
                   for k in frame_idx],
        )],
    )
    return _apply_theme(fig, title)


# --------------------------------------------------------------------------
# 2. Species-richness-per-island timeseries
# --------------------------------------------------------------------------
def richness_timeseries(rows, names=None, *, title="Species richness per island"):
    names = names or list(rows[0]["islands"].keys())
    times = history_times(rows)
    fig = go.Figure()
    for i, n in enumerate(names):
        series = [richness_of(r["islands"][n]) for r in rows]
        fig.add_trace(go.Scatter(
            x=times, y=series, mode="lines", name=n,
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2),
            hovertemplate=f"<b>{n}</b><br>gen=%{{x:.0f}}<br>richness=%{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title="generation")
    fig.update_yaxes(title="number of species present", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return _apply_theme(fig, title)


# --------------------------------------------------------------------------
# 3. Per-island abundance stacked area (top-K species + Other)
# --------------------------------------------------------------------------
def abundance_stacked_area(rows, island_name, *, top_k=7,
                           title=None):
    title = title or f"Species abundance dynamics — {island_name}"
    times = history_times(rows)
    species = list(rows[0]["islands"][island_name].keys())
    peak = {s: max(r["islands"][island_name].get(s, 0.0) for r in rows) for s in species}
    top = [s for s, _ in sorted(peak.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]

    fig = go.Figure()
    for i, s in enumerate(top):
        series = [r["islands"][island_name].get(s, 0.0) for r in rows]
        fig.add_trace(go.Scatter(
            x=times, y=series, mode="lines", name=s, stackgroup="one",
            line=dict(width=0.5, color=CATEGORICAL[i % len(CATEGORICAL)]),
            fillcolor=CATEGORICAL[i % len(CATEGORICAL)],
            hovertemplate=f"<b>{s}</b><br>gen=%{{x:.0f}}<br>count=%{{y:.0f}}<extra></extra>",
        ))
    # Other = everything outside top-k
    other = [sum(c for sp, c in r["islands"][island_name].items() if sp not in top)
             for r in rows]
    if any(o > 0 for o in other):
        fig.add_trace(go.Scatter(
            x=times, y=other, mode="lines", name="Other", stackgroup="one",
            line=dict(width=0.5, color=MUTED), fillcolor=MUTED,
            hovertemplate="<b>Other species</b><br>gen=%{x:.0f}<br>count=%{y:.0f}<extra></extra>",
        ))
    fig.update_xaxes(title="generation")
    fig.update_yaxes(title="individuals", rangemode="tozero")
    return _apply_theme(fig, title)


# --------------------------------------------------------------------------
# 4. Scatter: an island metric vs area or distance (study charts)
# --------------------------------------------------------------------------
def metric_vs_predictor(xvals, yvals, labels, *, xtitle, ytitle, title,
                        logx=False, trend=True):
    x = np.asarray(xvals, dtype=float)
    y = np.asarray(yvals, dtype=float)
    fig = go.Figure()
    # Markers only (no always-on text): island labels collided badly where
    # several islands share a similar area. Identity lives in the hover.
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", text=labels,
        marker=dict(size=14, color=CATEGORICAL[0], line=dict(color=SURFACE, width=2)),
        name="islands",
        hovertemplate="<b>%{text}</b><br>" + xtitle + "=%{x:.1f}<br>" +
                      ytitle + "=%{y:.2f}<extra></extra>",
    ))
    if trend and len(x) >= 2:
        xx = np.log(x) if logx else x
        if np.ptp(xx) > 0:
            a, b = np.polyfit(xx, y, 1)
            order = np.argsort(xx)
            fig.add_trace(go.Scatter(
                x=x[order], y=(a * xx + b)[order], mode="lines",
                line=dict(color=CATEGORICAL[1], width=2, dash="dash"),
                name="linear fit",
                hovertemplate="fit<extra></extra>",
            ))
    fig.update_xaxes(title=xtitle, type="log" if logx else "linear")
    fig.update_yaxes(title=ytitle, rangemode="tozero")
    return _apply_theme(fig, title)


# --------------------------------------------------------------------------
# 5. Migration network over the spatial map (edge width ~ 1/(1+distance))
# --------------------------------------------------------------------------
def migration_network(island_meta, *, title="Inter-island migration kernel  1/(1+distance)"):
    names = list(island_meta.keys())
    xs = {n: island_meta[n]["x"] for n in names}
    ys = {n: island_meta[n]["y"] for n in names}
    areas = [island_meta[n]["area"] for n in names]

    fig = go.Figure()
    # edges
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = math.hypot(xs[a] - xs[b], ys[a] - ys[b])
            rate = 1.0 / (1.0 + d)
            fig.add_trace(go.Scatter(
                x=[xs[a], xs[b]], y=[ys[a], ys[b]], mode="lines",
                line=dict(color="rgba(42,120,214,0.45)", width=max(0.4, 60.0 * rate)),
                hoverinfo="text",
                text=f"{a} ↔ {b}<br>distance={d:.1f}<br>rate={rate:.3f}",
                showlegend=False,
            ))
    sizes = [max(10.0, 2.6 * math.sqrt(a / math.pi) + 8.0) for a in areas]
    fig.add_trace(go.Scatter(
        x=[xs[n] for n in names], y=[ys[n] for n in names],
        mode="markers+text", text=names, textposition="top center",
        textfont=dict(color=INK_SECONDARY, size=11),
        marker=dict(size=sizes, color=CATEGORICAL[0], line=dict(color=SURFACE, width=2)),
        showlegend=False,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(height=560)
    fig.update_xaxes(title="x position", constrain="domain")
    fig.update_yaxes(title="y position", scaleanchor="x", scaleratio=1, constrain="domain")
    return _apply_theme(fig, title)
