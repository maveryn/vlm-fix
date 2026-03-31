#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.model_registry import MODEL_REGISTRY  # noqa: E402


def _safe_model_name(name: str) -> str:
    return name.replace("/", "_")


def _split_labels(s: str) -> List[str]:
    labels = [x.strip() for x in str(s).split("|") if x.strip()]
    if len(labels) != 2:
        raise ValueError(f"Expected 2 labels in valid_labels, got: {s}")
    return labels


def _extract_answer(text: str, labels: List[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    label_map = {lbl.lower(): lbl for lbl in labels}

    exact = label_map.get(raw.lower())
    if exact is not None:
        return exact

    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    for candidate in reversed(boxed):
        tok = candidate.strip()
        if tok.lower() in label_map:
            return label_map[tok.lower()]

    low = raw.lower()
    matches: List[tuple[int, str]] = []
    for lbl in labels:
        patt = r"\b" + re.escape(lbl.lower()) + r"\b"
        for m in re.finditer(patt, low):
            matches.append((m.start(), lbl))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]

    one_char = {lbl.upper(): lbl for lbl in labels if len(lbl) == 1 and lbl.isalpha()}
    if one_char:
        for tok in reversed(re.findall(r"[A-Za-z]", raw)):
            key = tok.upper()
            if key in one_char:
                return one_char[key]

    return ""


def _build_message(row: pd.Series, dataset_dir: Path) -> List[Dict[str, str]]:
    image_abs = str((dataset_dir / str(row["image_path"])).resolve())
    text = str(row["prompt"])
    if str(row.get("image_text_order", "image_first")) == "text_first":
        return [{"type": "text", "value": text}, {"type": "image", "value": image_abs}]
    return [{"type": "image", "value": image_abs}, {"type": "text", "value": text}]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM-Fix benchmark eval")
    parser.add_argument("--dataset-dir", type=str, default="data/generated/vlm_fix")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--games", nargs="+", default=["tictactoe", "reversi", "connect4", "dots_boxes"])
    parser.add_argument("--prompt-types", nargs="+", default=["direct"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--runs-dir", type=str, default="runs/vlm_fix")
    parser.add_argument("--results-dir", type=str, default="results/vlm_fix")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def _run_single(model_name: str, game: str, instances: pd.DataFrame, dataset_dir: Path, out_jsonl: Path, batch_size: int, max_new_tokens: int) -> pd.DataFrame:
    ctor = MODEL_REGISTRY[model_name]
    model = ctor(vllm_batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=0.0)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, object]] = []

    with out_jsonl.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(instances), batch_size), desc=f"{game} | {model_name}", unit="batch"):
            chunk = instances.iloc[start : start + batch_size]
            messages = [_build_message(row, dataset_dir=dataset_dir) for _, row in chunk.iterrows()]
            preds, metas = model.generate_batch_with_meta(messages, dataset="vlm_fix")

            for (_, src), pred, meta in zip(chunk.iterrows(), preds, metas):
                labels = _split_labels(str(src["valid_labels"]))
                extracted = _extract_answer(str(pred), labels)
                answer = str(src["answer"])
                rec = {
                    "model": model_name,
                    "game": str(src["game"]),
                    "index": int(src["index"]),
                    "state_id": int(src["state_id"]),
                    "render_variant": str(src["render_variant"]),
                    "rule_variant": str(src["rule_variant"]),
                    "prompt_variant": str(src.get("prompt_variant", "original")),
                    "prompt_type": str(src["prompt_type"]),
                    "question_target": str(src["question_target"]),
                    "image_text_order": str(src["image_text_order"]),
                    "prompt": str(src["prompt"]),
                    "valid_labels": str(src["valid_labels"]),
                    "answer": answer,
                    "prediction": str(pred),
                    "extracted_answer": extracted,
                    "correct": bool(extracted == answer),
                    "image_path": str(src["image_path"]),
                    "output_tokens": (meta or {}).get("output_tokens") if isinstance(meta, dict) else None,
                    "finish_reason": (meta or {}).get("finish_reason") if isinstance(meta, dict) else None,
                }
                rows_out.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return pd.DataFrame(rows_out)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "game",
        "render_variant",
        "rule_variant",
        "prompt_variant",
        "prompt_type",
        "question_target",
        "image_text_order",
    ]
    return (
        df.groupby(keys, dropna=False)["correct"]
        .agg(n="size", accuracy="mean")
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir)
    instances_path = dataset_dir / "instances.parquet"
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing dataset: {instances_path}")

    df = pd.read_parquet(instances_path).copy()
    if "index" not in df.columns:
        df.insert(0, "index", range(1, len(df) + 1))

    df = df[df["game"].isin(args.games)].copy()
    df = df[df["prompt_type"].isin(args.prompt_types)].copy()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, object]] = []

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")
        for game in args.games:
            game_df = df[df["game"] == game].copy()
            if game_df.empty:
                continue
            out_jsonl = runs_dir / game / f"{_safe_model_name(model_name)}__prompt-{'-'.join(args.prompt_types)}.jsonl"

            if args.skip_existing and out_jsonl.exists():
                loaded = pd.read_json(out_jsonl, lines=True)
                all_frames.append(loaded)
                run_rows.append(
                    {
                        "model": model_name,
                        "game": game,
                        "status": "skipped_existing",
                        "n": int(len(loaded)),
                        "run_file": str(out_jsonl),
                    }
                )
                print(f"[skip] model={model_name} game={game} n={len(loaded)} file={out_jsonl}")
                continue

            print(f"[run] model={model_name} game={game} n={len(game_df)}")
            pred_df = _run_single(
                model_name=model_name,
                game=game,
                instances=game_df,
                dataset_dir=dataset_dir,
                out_jsonl=out_jsonl,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            all_frames.append(pred_df)
            combo_acc = float(pred_df["correct"].mean()) if len(pred_df) else 0.0
            run_rows.append(
                {
                    "model": model_name,
                    "game": game,
                    "status": "ok",
                    "n": int(len(pred_df)),
                    "accuracy": combo_acc,
                    "run_file": str(out_jsonl),
                }
            )
            print(f"[done] model={model_name} game={game} n={len(pred_df)} acc={combo_acc:.4f} file={out_jsonl}")

    if not all_frames:
        print("[warn] no runs produced")
        return 0

    pred_all = pd.concat(all_frames, ignore_index=True)
    pred_path = results_dir / "predictions_all.parquet"
    pred_all.to_parquet(pred_path, index=False)

    summary = _summarize(pred_all)
    summary_path = results_dir / "summary_full_combo.csv"
    summary.to_csv(summary_path, index=False)

    simple = (
        pred_all.groupby(["model", "game", "rule_variant"], as_index=False)["correct"]
        .agg(n="size", accuracy="mean")
        .sort_values(["model", "game", "rule_variant"])
        .reset_index(drop=True)
    )
    simple_path = results_dir / "summary_by_game_rule.csv"
    simple.to_csv(simple_path, index=False)

    runlog = pd.DataFrame(run_rows)
    runlog_path = results_dir / "run_log.csv"
    runlog.to_csv(runlog_path, index=False)

    print(f"[ok] predictions: {pred_path}")
    print(f"[ok] summary full: {summary_path}")
    print(f"[ok] summary game_rule: {simple_path}")
    print(f"[ok] run log: {runlog_path}")
    print(simple.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
