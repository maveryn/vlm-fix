from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


_RE_BOXED = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}", re.IGNORECASE)
_RE_CURLY = re.compile(r"\{\s*(-?\d+)\s*\}")
_RE_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(-?\d+)", re.IGNORECASE)
_RE_INT = re.compile(r"-?\d+")


def _extract_reference_number(text: str) -> str:
    s = str(text or "").strip()
    m = _RE_INT.search(s)
    return m.group(0) if m else ""


def _extract_numeric_answer(text: str) -> str:
    """Extract a single numeric answer from model output.

    Priority:
    1) \\boxed{n}
    2) any {n} (last match)
    3) explicit 'answer: n' (last match)
    4) final integer in text
    """
    s = str(text or "").strip()
    if not s:
        return ""

    boxed = _RE_BOXED.findall(s)
    if boxed:
        return boxed[-1]

    curly = _RE_CURLY.findall(s)
    if curly:
        return curly[-1]

    answer_tags = _RE_ANSWER.findall(s)
    if answer_tags:
        return answer_tags[-1]

    ints = _RE_INT.findall(s)
    if ints:
        return ints[-1]

    return ""


def _score_row(pred: str, ground_truth: str, expected_bias: str) -> Dict[str, float]:
    pred_norm = _extract_numeric_answer(pred)
    gt_norm = _extract_reference_number(ground_truth)
    bias_norm = _extract_reference_number(expected_bias)

    is_correct = bool(pred_norm and gt_norm and pred_norm == gt_norm)
    matches_bias = bool(pred_norm and bias_norm and pred_norm == bias_norm)

    return {
        "prediction_norm": pred_norm,
        "accuracy": float(is_correct),
        "bias_ratio": float(matches_bias),
    }


def add_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pred_df.iterrows():
        scored = _score_row(
            pred=str(row.get("prediction", "")),
            ground_truth=str(row.get("ground_truth", "")),
            expected_bias=str(row.get("expected_bias", "")),
        )
        rows.append(scored)

    score_df = pd.DataFrame(rows)
    out = pred_df.reset_index(drop=True).copy()
    out["prediction_norm"] = score_df["prediction_norm"]
    out["accuracy"] = score_df["accuracy"]
    out["bias_ratio"] = score_df["bias_ratio"]
    return out


def aggregate(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = (
        pred_df.groupby(["model", "topic", "prompt_style"], as_index=False)
        .agg(
            n_total=("accuracy", "size"),
            n_correct=("accuracy", "sum"),
            accuracy=("accuracy", "mean"),
            bias_ratio=("bias_ratio", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
        )
        .sort_values(["model", "topic", "prompt_style"])
    )
    summary["n_total"] = summary["n_total"].astype(int)
    summary["n_correct"] = summary["n_correct"].round().astype(int)

    delta_rows = []
    for (model_name, topic), grp in summary.groupby(["model", "topic"], as_index=False):
        base = grp[grp["prompt_style"] == "original"]
        base_acc = float(base.iloc[0]["accuracy"]) if not base.empty else None
        base_bias = float(base.iloc[0]["bias_ratio"]) if not base.empty else None
        base_total = int(base.iloc[0]["n_total"]) if not base.empty else None
        base_correct = int(base.iloc[0]["n_correct"]) if not base.empty else None

        for _, row in grp.iterrows():
            delta_rows.append(
                {
                    "model": model_name,
                    "topic": topic,
                    "prompt_style": str(row["prompt_style"]),
                    "n_total": int(row["n_total"]),
                    "n_correct": int(row["n_correct"]),
                    "accuracy": float(row["accuracy"]),
                    "bias_ratio": float(row["bias_ratio"]),
                    "avg_output_tokens": float(row["avg_output_tokens"]) if pd.notna(row["avg_output_tokens"]) else None,
                    "n_total_original": base_total,
                    "n_correct_original": base_correct,
                    "accuracy_original": base_acc,
                    "bias_ratio_original": base_bias,
                    "accuracy_delta_vs_original": (float(row["accuracy"]) - base_acc) if base_acc is not None else None,
                    "bias_delta_vs_original": (float(row["bias_ratio"]) - base_bias) if base_bias is not None else None,
                }
            )

    deltas = pd.DataFrame(delta_rows).sort_values(["model", "topic", "prompt_style"])

    payload = {
        "n_predictions": int(len(pred_df)),
        "overall_accuracy": float(pred_df["accuracy"].mean()) if len(pred_df) else 0.0,
        "overall_bias_ratio": float(pred_df["bias_ratio"].mean()) if len(pred_df) else 0.0,
        "models": sorted(pred_df["model"].astype(str).unique().tolist()) if len(pred_df) else [],
        "topics": sorted(pred_df["topic"].astype(str).unique().tolist()) if len(pred_df) else [],
        "prompt_styles": sorted(pred_df["prompt_style"].astype(str).unique().tolist()) if len(pred_df) else [],
    }

    return summary, deltas, payload


def save_reports(
    out_dir: str | Path,
    pred_df: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    payload: dict,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_df.to_parquet(out_path / "all_predictions.parquet", index=False)
    summary.to_csv(out_path / "summary_by_model_topic_style.csv", index=False)
    deltas.to_csv(out_path / "summary_deltas_vs_original.csv", index=False)

    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
