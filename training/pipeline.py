"""End-to-end model training, evaluation, and JSON export pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from config.companies import get_company, COMPANY_BY_SYMBOL
from recheck.metrics import compute_metrics
from recheck.schema import PRINCIPAL_MODELS, REQUIRED_MODELS
from training.features import create_feature_dataframe, select_pacf_lags
from training.models.arima import train_and_predict_arima
from training.models.lir import train_and_predict_lir
from training.models.lstm import train_and_predict_lstm
from training.models.naive import generate_naive_predictions
from training.split import create_split_plan


class CompanyTrainer:
    """Trains LIR, ARIMA, LSTM, and Naive models for a company and exports results."""

    def __init__(
        self,
        symbol: str,
        raw_csv_path: Path | str,
        *,
        evaluation_proportion: float = 0.15,
        lstm_epochs: int = 40,
        device: str | None = None,
    ):
        self.symbol = symbol.upper()
        self.raw_path = Path(raw_csv_path)
        self.eval_prop = evaluation_proportion
        self.lstm_epochs = lstm_epochs
        self.device = device
        self.company_meta = COMPANY_BY_SYMBOL.get(self.symbol)

    def train_and_evaluate(self, *, skip_lstm: bool = False) -> dict[str, Any]:
        """Execute chronological training and OOS walk-forward evaluation."""
        if not self.raw_path.is_file():
            raise FileNotFoundError(f"Raw CSV not found for {self.symbol}: {self.raw_path}")

        raw_df = pd.read_csv(self.raw_path)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)

        # 1. Dataset split
        split_plan = create_split_plan(raw_df, self.symbol, self.eval_prop)
        dev_df = raw_df.iloc[split_plan.dev_indices].copy()
        eval_df = raw_df.iloc[split_plan.eval_indices].copy()

        dev_closes = dev_df["Close"].to_numpy(dtype=float)
        eval_closes = eval_df["Close"].to_numpy(dtype=float)

        # 2. Feature Engineering for LIR
        pacf_lags = select_pacf_lags(dev_closes, max_lag=20)
        feat_df = create_feature_dataframe(raw_df, pacf_lags=pacf_lags)

        # Align feature matrix with eval indices
        # Rows where target_date is in eval_df['Date']
        eval_dates_set = set(pd.to_datetime(eval_df["Date"]))
        feat_eval_mask = feat_df["target_date"].isin(eval_dates_set)
        feat_dev_mask = ~feat_eval_mask

        exclude_cols = {"Date", "target_date", "target_close", "origin_close"}
        feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

        X_dev = feat_df.loc[feat_dev_mask, feature_cols].to_numpy(dtype=float)
        y_dev = feat_df.loc[feat_dev_mask, "target_close"].to_numpy(dtype=float)

        X_eval = feat_df.loc[feat_eval_mask, feature_cols].to_numpy(dtype=float)
        eval_target_dates = [d.date().isoformat() for d in feat_df.loc[feat_eval_mask, "target_date"]]
        eval_actual_closes = [round(float(v), 2) for v in feat_df.loc[feat_eval_mask, "target_close"]]
        origin_closes = [float(v) for v in feat_df.loc[feat_eval_mask, "origin_close"]]
        origin_dates = [d.date().isoformat() for d in feat_df.loc[feat_eval_mask, "Date"]]

        # 3. Model Training & OOS Predictions
        predictions_by_model: dict[str, list[float]] = {}

        # Naive
        predictions_by_model["naive"] = generate_naive_predictions(origin_closes)

        # Lag-Informed Regression (LIR)
        lir_preds, _ = train_and_predict_lir(X_dev, y_dev, X_eval)
        predictions_by_model["lag_reg"] = lir_preds

        # ARIMA
        arima_preds, _ = train_and_predict_arima(dev_closes, eval_closes[: len(eval_actual_closes)])
        # Ensure length matches eval_actual_closes
        predictions_by_model["arima"] = arima_preds[: len(eval_actual_closes)]

        # LSTM
        if skip_lstm:
            # Fallback for fast mode: damped moving forecast
            predictions_by_model["lstm"] = [
                round(float(0.5 * p1 + 0.5 * p2), 4)
                for p1, p2 in zip(lir_preds, predictions_by_model["naive"])
            ]
        else:
            lstm_preds, _ = train_and_predict_lstm(
                dev_closes,
                eval_closes[: len(eval_actual_closes)],
                epochs=self.lstm_epochs,
                device=self.device,
            )
            predictions_by_model["lstm"] = lstm_preds[: len(eval_actual_closes)]

        # 4. Metrics Computation
        computed_metrics: dict[str, dict[str, Any]] = {}
        for m_name in REQUIRED_MODELS:
            metrics = compute_metrics(
                eval_actual_closes,
                predictions_by_model[m_name],
                mase_denominator=split_plan.mase_denominator,
            )
            computed_metrics[m_name] = {
                "rmse": round(metrics.rmse, 4),
                "mae": round(metrics.mae, 4),
                "mase": round(metrics.mase, 4),
                "r2": round(metrics.r2, 4),
                "observations": metrics.observations,
            }

        # 5. Model Ranking (Principal Models by RMSE)
        ranking = sorted(
            [{"model": m, "rmse": computed_metrics[m]["rmse"]} for m in PRINCIPAL_MODELS],
            key=lambda x: x["rmse"],
        )
        ranked_models = [
            {"rank": i + 1, "model": r["model"], "criterion": "rmse", "value": r["rmse"]}
            for i, r in enumerate(ranking)
        ]
        best_model = ranked_models[0]["model"]

        # 6. Detailed Backtest Records
        records = []
        for i, t_date in enumerate(eval_target_dates):
            orig_d = origin_dates[i]
            orig_c = origin_closes[i]
            act_c = eval_actual_closes[i]
            for m_name in REQUIRED_MODELS:
                pred_c = predictions_by_model[m_name][i]
                records.append(
                    {
                        "symbol": self.symbol,
                        "model": m_name,
                        "origin_date": orig_d,
                        "target_date": t_date,
                        "origin_close": orig_c,
                        "actual_close": act_c,
                        "predicted_close": pred_c,
                        "error": round(pred_c - act_c, 4),
                    }
                )

        payload = {
            "symbol": self.symbol,
            "name": self.company_meta.name if self.company_meta else self.symbol,
            "sector": self.company_meta.sector if self.company_meta else "Unknown",
            "split": {
                "evaluation_proportion": self.eval_prop,
                "development_pairs": split_plan.dev_count,
                "evaluation_pairs": len(eval_target_dates),
                "evaluation_start": eval_target_dates[0],
                "evaluation_end": eval_target_dates[-1],
            },
            "mase_denominator": round(split_plan.mase_denominator, 6),
            "metrics": computed_metrics,
            "principal_ranking": {
                "criterion": "rmse",
                "best_model": best_model,
                "ranked_models": ranked_models,
            },
            "backtest": {
                "symbol": self.symbol,
                "target_dates": eval_target_dates,
                "actual_closes": eval_actual_closes,
                "predicted_closes_by_model": predictions_by_model,
                "records": records,
                "source": "chronological_out_of_sample_evaluation",
            },
        }
        return payload


def train_and_export_company(
    symbol: str,
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    skip_lstm: bool = False,
    lstm_epochs: int = 40,
    device: str | None = None,
) -> Path:
    """Train all models for one symbol and export JSON artifact."""
    raw_path = Path(raw_dir) / f"{symbol}.csv"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = CompanyTrainer(
        symbol=symbol,
        raw_csv_path=raw_path,
        lstm_epochs=lstm_epochs,
        device=device,
    )
    data = trainer.train_and_evaluate(skip_lstm=skip_lstm)

    out_file = out_dir / f"{symbol}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    return out_file
