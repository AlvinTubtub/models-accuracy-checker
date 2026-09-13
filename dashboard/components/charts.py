"""Plotly visualization helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette for models
MODEL_COLORS = {
    "Actual": "#111827",      # Deep dark
    "Naive": "#9CA3AF",       # Neutral grey
    "lag_reg": "#2563EB",     # Royal Blue
    "arima": "#059669",       # Emerald Green
    "lstm": "#D97706",        # Amber / Orange
    "Lag-Informed Regression": "#2563EB",
    "ARIMA": "#059669",
    "LSTM": "#D97706",
}


def plot_time_series(
    target_dates: list,
    actual_closes: list[float],
    predicted_by_model: dict[str, list[float]],
    symbol: str,
    name: str | None = None,
) -> go.Figure:
    """Interactive multi-model backtest price forecast line chart."""

    fig = go.Figure()

    # Actual Close
    fig.add_trace(
        go.Scatter(
            x=target_dates,
            y=actual_closes,
            mode="lines",
            name="Actual Close",
            line=dict(color=MODEL_COLORS["Actual"], width=2.5),
            hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> ₱%{y:.2f}<extra></extra>",
        )
    )

    # Models
    label_map = {
        "naive": "Naive Benchmark",
        "lag_reg": "Lag-Informed Regression (LIR)",
        "arima": "ARIMA",
        "lstm": "LSTM",
    }

    for model_key in ["naive", "lag_reg", "arima", "lstm"]:
        if model_key in predicted_by_model:
            color = MODEL_COLORS.get(model_key, "#6B7280")
            dash = "dash" if model_key == "naive" else "solid"
            width = 1.8 if model_key != "naive" else 1.5
            fig.add_trace(
                go.Scatter(
                    x=target_dates,
                    y=predicted_by_model[model_key],
                    mode="lines",
                    name=label_map.get(model_key, model_key),
                    line=dict(color=color, width=width, dash=dash),
                    hovertemplate=f"<b>{label_map.get(model_key, model_key)}:</b> ₱%{{y:.2f}}<extra></extra>",
                )
            )

    title_text = f"{symbol} — Out-of-Sample Price Forecasts vs. Actuals"
    if name:
        title_text += f" ({name})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis=dict(title="Evaluation Target Date", showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Closing Price (PHP)", showgrid=True, gridcolor="#E5E7EB"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        height=500,
    )
    return fig


def plot_residuals(
    target_dates: list,
    actual_closes: list[float],
    predicted_by_model: dict[str, list[float]],
    symbol: str,
) -> go.Figure:
    """Forecast error / residual plot over time (Predicted - Actual)."""

    fig = go.Figure()

    # Zero error baseline
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#4B5563",
        annotation_text="Zero Error Line",
        annotation_position="bottom right",
    )

    label_map = {
        "naive": "Naive",
        "lag_reg": "LIR",
        "arima": "ARIMA",
        "lstm": "LSTM",
    }

    for model_key in ["naive", "lag_reg", "arima", "lstm"]:
        if model_key in predicted_by_model:
            preds = predicted_by_model[model_key]
            residuals = [p - a for p, a in zip(preds, actual_closes)]
            color = MODEL_COLORS.get(model_key, "#6B7280")
            fig.add_trace(
                go.Scatter(
                    x=target_dates,
                    y=residuals,
                    mode="lines",
                    name=label_map.get(model_key, model_key),
                    line=dict(color=color, width=1.4),
                    hovertemplate=f"<b>{label_map.get(model_key, model_key)} Error:</b> ₱%{{y:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=f"{symbol} — Prediction Residuals (Predicted - Actual Close)", font=dict(size=15)),
        xaxis=dict(title="Evaluation Date", showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Error (PHP)", showgrid=True, gridcolor="#E5E7EB"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        height=400,
    )
    return fig


def plot_win_rates(win_summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart illustrating model win rates against the Naive baseline."""

    fig = go.Figure()
    if win_summary_df.empty:
        return fig

    colors = [
        "#10B981" if rate >= 50.0 else "#F59E0B"
        for rate in win_summary_df["win_rate_pct"]
    ]

    fig.add_trace(
        go.Bar(
            x=win_summary_df["model_label"],
            y=win_summary_df["win_rate_pct"],
            text=[f"{r:.1f}% ({int(c)}/{int(t)})" for r, c, t in zip(
                win_summary_df["win_rate_pct"],
                win_summary_df["companies_beating_naive"],
                win_summary_df["total_companies"],
            )],
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
            hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=50.0,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="50% Benchmark Threshold",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text="Model Win Rate vs. Naive Benchmark (% Companies where MASE < 1.0)", font=dict(size=15)),
        yaxis=dict(title="Win Rate (%)", range=[0, 110], showgrid=True, gridcolor="#E5E7EB"),
        xaxis=dict(title="Forecasting Model"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig


def plot_metric_distributions(df: pd.DataFrame, metric: str = "recomputed_mase") -> go.Figure:
    """Box plot of metric distribution across all companies."""

    principal = df[df["is_principal"]].copy()
    if principal.empty:
        return go.Figure()

    metric_names = {
        "recomputed_mase": "MASE",
        "recomputed_rmse": "RMSE",
        "recomputed_mae": "MAE",
        "recomputed_r2": "R²",
    }
    metric_label = metric_names.get(metric, metric)

    fig = px.box(
        principal,
        x="model_label",
        y=metric,
        color="model_label",
        color_discrete_map=MODEL_COLORS,
        points="all",
        hover_data=["symbol", "sector"],
        labels={"model_label": "Model", metric: metric_label},
        title=f"Cross-Company Distribution of {metric_label}",
    )

    if metric == "recomputed_mase":
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text="MASE = 1.0 (Naive Line)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig


def plot_actual_vs_predicted(
    actual: list[float],
    predicted: list[float],
    model_name: str,
) -> go.Figure:
    """Parity scatter plot (Actual vs. Predicted)."""

    fig = go.Figure()

    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal 45° Line (y = x)",
            line=dict(color="#9CA3AF", dash="dash"),
        )
    )

    color = MODEL_COLORS.get(model_name, "#2563EB")
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            name=f"{model_name} Points",
            marker=dict(color=color, size=6, opacity=0.7),
            hovertemplate="Actual: ₱%{x:.2f}<br>Predicted: ₱%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Actual vs. Predicted Close ({model_name})", font=dict(size=15)),
        xaxis=dict(title="Actual Close (PHP)", showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Predicted Close (PHP)", showgrid=True, gridcolor="#E5E7EB"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig
