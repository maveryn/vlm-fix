#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.model_registry import MODEL_REGISTRY  # noqa: E402

GAMES = ["tictactoe", "reversi", "connect4", "dots_boxes"]
RULES = ["standard", "inverse"]
RUN_RE = re.compile(r"^(?P<safe>.+)__(?P<tag>[^.]+)\.jsonl$")

MODEL_ORDER = [
    "gpt-4.1",
    "gpt-5.2",
    "claude-sonnet-4-0",
    "claude-sonnet-4-5",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-14B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "allenai/Molmo2-4B",
    "allenai/Molmo2-8B",
]

DISPLAY_NAME_MAP = {
    "gpt-4.1": "GPT-4.1",
    "gpt-5.2": "GPT-5.2",
    "claude-sonnet-4-0": "Sonnet-4.0",
    "claude-sonnet-4-5": "Sonnet-4.5",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen/Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
    "OpenGVLab/InternVL3_5-4B": "InternVL3.5-4B",
    "OpenGVLab/InternVL3_5-8B": "InternVL3.5-8B",
    "OpenGVLab/InternVL3_5-14B": "InternVL3.5-14B",
    "allenai/Molmo2-4B": "Molmo-4B",
    "allenai/Molmo2-8B": "Molmo-8B",
}


def _safe_to_model_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for k in MODEL_REGISTRY.keys():
        safe = k.replace("/", "_")
        if safe not in out or ("/" in k and "/" not in out[safe]):
            out[safe] = k
    return out


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_candidates(runs_dir: Path, run_tag: str) -> dict[tuple[str, str], tuple[Path, int]]:
    chosen: dict[tuple[str, str], tuple[Path, int]] = {}
    required_rules = set(RULES)
    required_targets = {"winner", "loser"}

    for game in GAMES:
        game_dir = runs_dir / game
        if not game_dir.exists():
            continue
        for fp in game_dir.glob("*.jsonl"):
            m = RUN_RE.match(fp.name)
            if not m:
                continue
            safe = m.group("safe")
            tag = m.group("tag")
            if tag != run_tag:
                continue
            rows = _load_jsonl(fp)
            if not rows:
                continue
            rules = {str(r.get("rule_variant", "")) for r in rows}
            targets = {str(r.get("question_target", "")) for r in rows}
            if not required_rules.issubset(rules):
                continue
            if not required_targets.issubset(targets):
                continue
            key = (safe, game)
            n = len(rows)
            prev = chosen.get(key)
            if prev is None or n > prev[1]:
                chosen[key] = (fp, n)
    return chosen


def _select_complete_models(cands: dict[tuple[str, str], tuple[Path, int]]) -> list[str]:
    models = sorted({safe for safe, _ in cands.keys()})
    return [m for m in models if all((m, g) in cands for g in GAMES)]


def _exclude_models(complete_models: list[str], exclude_models: list[str]) -> list[str]:
    if not exclude_models:
        return complete_models
    excludes = {m.strip() for m in exclude_models if m and m.strip()}
    if not excludes:
        return complete_models
    safe_map = _safe_to_model_map()
    kept: list[str] = []
    for safe in complete_models:
        model_name = safe_map.get(safe, safe)
        if safe in excludes or model_name in excludes:
            continue
        kept.append(safe)
    return kept


def _build_long_df(cands: dict[tuple[str, str], tuple[Path, int]], models: list[str]) -> pd.DataFrame:
    safe_map = _safe_to_model_map()
    out_rows: list[dict] = []
    for safe in models:
        model_name = safe_map.get(safe, safe)
        for game in GAMES:
            fp, n = cands[(safe, game)]
            rows = _load_jsonl(fp)
            for r in rows:
                out_rows.append(
                    {
                        "model": model_name,
                        "safe_model": safe,
                        "game": game,
                        "rule_variant": str(r.get("rule_variant", "")),
                        "question_target": str(r.get("question_target", "")),
                        "correct": 1 if bool(r.get("correct", False)) else 0,
                        "run_file": str(fp),
                        "run_rows": n,
                    }
                )
    return pd.DataFrame(out_rows)


def _summarize_by_rule(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby(["model", "safe_model", "game", "rule_variant"], as_index=False)["correct"]
        .agg(n="size", accuracy="mean")
        .sort_values(["model", "game", "rule_variant"])
        .reset_index(drop=True)
    )


def _ordered_models(models: list[str]) -> list[str]:
    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order_idx.get(m, 10_000), m))


def _to_display_name(model: str) -> str:
    return DISPLAY_NAME_MAP.get(model, model.split("/")[-1])


def _build_sheet(summary_df: pd.DataFrame, game: str, model_order: list[str]) -> pd.DataFrame:
    g = summary_df[summary_df["game"] == game].copy()
    out = pd.DataFrame({"Model": model_order})
    out["TextOnly_Std"] = out["Model"].map(
        dict(zip(g[g["rule_variant"] == "standard"]["model"], g[g["rule_variant"] == "standard"]["accuracy"]))
    ) * 100.0
    out["TextOnly_Inv"] = out["Model"].map(
        dict(zip(g[g["rule_variant"] == "inverse"]["model"], g[g["rule_variant"] == "inverse"]["accuracy"]))
    ) * 100.0
    out["Gap_StdMinusInv"] = out["TextOnly_Std"] - out["TextOnly_Inv"]
    out["Model"] = out["Model"].map(_to_display_name)
    return out


def _append_overall_avg_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_cols = [c for c in out.columns if c != "Model"]
    avg_row = {"Model": "Overall Avg"}
    for c in metric_cols:
        avg_row[c] = out[c].mean()
    return pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)


def _build_average_sheet(summary_df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    avg = pd.DataFrame({"Model": model_order})
    rows = []
    for model in model_order:
        sub = summary_df[summary_df["model"] == model]
        std = float(sub[sub["rule_variant"] == "standard"]["accuracy"].mean() * 100.0)
        inv = float(sub[sub["rule_variant"] == "inverse"]["accuracy"].mean() * 100.0)
        rows.append({"Model": _to_display_name(model), "TextOnly_Std": std, "TextOnly_Inv": inv, "Gap_StdMinusInv": std - inv})
    return pd.DataFrame(rows)


def _write_excel(summary_df: pd.DataFrame, out_xlsx: Path) -> None:
    model_order = _ordered_models(summary_df["model"].unique().tolist())
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for game in GAMES:
            sheet = _append_overall_avg_row(_build_sheet(summary_df, game, model_order))
            sheet.to_excel(writer, sheet_name=game, index=False)
        avg_sheet = _append_overall_avg_row(_build_average_sheet(summary_df, model_order))
        avg_sheet.to_excel(writer, sheet_name="average", index=False)

        for ws in writer.book.worksheets:
            ws.column_dimensions["A"].width = 34
            for cell in ws[1]:
                cell.alignment = Alignment(horizontal="center")
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    if cell.value is not None:
                        cell.number_format = "0.0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build text-only VLM-Fix summary workbook from run jsonl files.")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/vlm_fix_text_only"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/vlm_fix_text_only"))
    ap.add_argument("--run-tag", type=str, default="prompt-direct")
    ap.add_argument("--exclude-models", nargs="*", default=[])
    args = ap.parse_args()

    cands = _collect_candidates(args.runs_dir, run_tag=args.run_tag)
    complete_models = _select_complete_models(cands)
    complete_models = _exclude_models(complete_models, args.exclude_models)
    if not complete_models:
        raise RuntimeError(f"No complete models found for run_tag={args.run_tag} in {args.runs_dir}")

    long_df = _build_long_df(cands, complete_models)
    summary_df = _summarize_by_rule(long_df)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.results_dir / f"predictions_long__{args.run_tag}.csv"
    summary_path = args.results_dir / f"summary_by_game_rule__{args.run_tag}.csv"
    excel_path = args.results_dir / "vlm_fix_text_only_summary_2x_direct.xlsx"
    model_path = args.results_dir / f"models_included__{args.run_tag}.txt"

    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _write_excel(summary_df, excel_path)
    model_path.write_text("\n".join(_ordered_models(summary_df["model"].unique().tolist())) + "\n", encoding="utf-8")

    print(f"[ok] models={summary_df['model'].nunique()} rows={len(summary_df)}")
    print(f"[ok] {summary_path}")
    print(f"[ok] {excel_path}")


if __name__ == "__main__":
    main()
