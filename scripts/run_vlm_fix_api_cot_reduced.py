#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_vlm_fix_matrix as matrix  # noqa: E402


API_MODELS = [
    "gpt-4.1",
    "gpt-5.2",
    "claude-sonnet-4-0",
    "claude-sonnet-4-5",
]

GAMES = ["tictactoe", "reversi", "connect4", "dots_boxes"]
PROMPT_VARIANTS = ["standard", "tag", "tag_sem"]
RULE_VARIANTS = ["standard", "inverse"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reduced VLM-Fix CoT evaluation for API models using the shared "
            "300-state-per-rule seed set from the existing reduced direct runs."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default="data/generated/vlm_fix")
    parser.add_argument("--models", nargs="+", default=API_MODELS)
    parser.add_argument("--games", nargs="+", default=GAMES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--runs-dir", type=str, default="runs/vlm_fix")
    parser.add_argument("--results-dir", type=str, default="results/vlm_fix")
    parser.add_argument("--run-tag", type=str, default="prompt-cot-reduced300")
    parser.add_argument("--seed-reference-model", type=str, default="gpt-4.1")
    parser.add_argument("--seed-reference-run-tag", type=str, default="prompt-direct-reduced300")
    parser.add_argument("--quiet", dest="quiet", action="store_true", default=True)
    parser.add_argument("--no-quiet", dest="quiet", action="store_false")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_seed_keep_set(
    runs_dir: Path,
    game: str,
    ref_model: str,
    ref_run_tag: str,
) -> Set[Tuple[str, int]]:
    seed_path = runs_dir / game / f"{matrix._safe_model_name(ref_model)}__{ref_run_tag}.jsonl"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing reference reduced run: {seed_path}")

    keep: Set[Tuple[str, int]] = set()
    with seed_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("prompt_type")) != "direct":
                continue
            if str(row.get("question_target")) != "winner":
                continue
            if str(row.get("image_text_order")) != "image_first":
                continue
            if str(row.get("render_variant")) != "canonical":
                continue
            if str(row.get("prompt_variant")) != "standard":
                continue
            keep.add((str(row["rule_variant"]), int(row["state_id"])))

    if len(keep) != 600:
        raise ValueError(
            f"Expected 600 seed rows (300 per rule) for {game}, got {len(keep)} from {seed_path}"
        )
    return keep


def _build_reduced_cot_slice(
    dataset_dir: Path,
    runs_dir: Path,
    games: List[str],
    ref_model: str,
    ref_run_tag: str,
) -> pd.DataFrame:
    instances_path = dataset_dir / "instances.parquet"
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing dataset: {instances_path}")

    df = pd.read_parquet(instances_path).copy()
    if "index" not in df.columns:
        df.insert(0, "index", range(1, len(df) + 1))

    df = df[df["game"].isin(games)].copy()
    df = df[df["prompt_type"] == "cot"].copy()
    df = df[df["render_variant"] == "canonical"].copy()
    df = df[df["prompt_variant"].isin(PROMPT_VARIANTS)].copy()
    df = df[df["rule_variant"].isin(RULE_VARIANTS)].copy()
    df = df[df["image_text_order"] == "image_first"].copy()
    df = df[df["question_target"] == "winner"].copy()

    keep_by_game: Dict[str, Set[Tuple[str, int]]] = {
        game: _load_seed_keep_set(runs_dir, game, ref_model, ref_run_tag) for game in games
    }

    df = df[
        df.apply(
            lambda row: (str(row["rule_variant"]), int(row["state_id"])) in keep_by_game[str(row["game"])],
            axis=1,
        )
    ].copy()

    expected = {game: 1800 for game in games}
    actual = df.groupby("game").size().to_dict()
    if actual != expected:
        raise ValueError(f"Unexpected reduced CoT slice counts: expected {expected}, got {actual}")

    return df


def main() -> int:
    args = _parse_args()
    matrix._configure_quiet_mode(bool(args.quiet))

    dataset_dir = Path(args.dataset_dir)
    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in matrix.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")

    df = _build_reduced_cot_slice(
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        games=list(args.games),
        ref_model=str(args.seed_reference_model),
        ref_run_tag=str(args.seed_reference_run_tag),
    )

    per_game = df.groupby("game").size().to_dict()
    per_variant = (
        df.groupby(["game", "prompt_variant", "rule_variant"]).size().sort_index().to_dict()
    )
    print(f"[filter] reduced CoT rows: total={len(df)} | per_game={per_game}")
    print(f"[filter] per_game_variant_rule={per_variant}")

    if args.dry_run:
        print("[dry-run] reduced CoT slice built successfully")
        return 0

    combos: List[Tuple[str, str]] = []
    for model_name in args.models:
        for game in args.games:
            if not df[df["game"] == game].empty:
                combos.append((model_name, game))

    total = len(combos)
    if total == 0:
        raise ValueError("No model/game combinations to run.")

    all_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, object]] = []
    done = 0

    combo_bar = tqdm(total=total, desc="api cot combinations", unit="combo")
    for i, (model_name, game) in enumerate(combos, start=1):
        game_df = df[df["game"] == game].copy()
        out_jsonl = runs_dir / game / f"{matrix._safe_model_name(model_name)}__{args.run_tag}.jsonl"
        combo_bar.set_postfix_str(f"{model_name} | {game}")

        print(f"[combo {i}/{total}] model={model_name} game={game} n={len(game_df)}")

        if args.skip_existing and out_jsonl.exists():
            loaded = pd.read_json(out_jsonl, lines=True)
            if len(loaded) == len(game_df):
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
                done += 1
                print(
                    f"[done {done}/{total}] skipped_existing acc={float(loaded['correct'].mean()):.4f} file={out_jsonl}"
                )
                combo_bar.update(1)
                continue
            print(
                f"[warn] existing file length mismatch ({len(loaded)} != {len(game_df)}), rerunning: {out_jsonl}"
            )

        pred_df = matrix._run_single(
            model_name=model_name,
            game=game,
            instances=game_df,
            dataset_dir=dataset_dir,
            out_jsonl=out_jsonl,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            combo_i=i,
            combo_total=total,
            quiet=bool(args.quiet),
        )
        combo_acc = float(pred_df["correct"].mean()) if len(pred_df) else 0.0
        all_frames.append(pred_df)
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
        done += 1
        print(f"[done {done}/{total}] acc={combo_acc:.4f} file={out_jsonl}")
        combo_bar.update(1)

        gc.collect()
    combo_bar.close()

    pred_all = pd.concat(all_frames, ignore_index=True)
    pred_path = results_dir / f"predictions_all__{args.run_tag}.parquet"
    pred_all.to_parquet(pred_path, index=False)

    summary = matrix._summarize(pred_all)
    summary_path = results_dir / f"summary_full_combo__{args.run_tag}.csv"
    summary.to_csv(summary_path, index=False)

    by_rule = (
        pred_all.groupby(["model", "game", "rule_variant"], as_index=False)["correct"]
        .agg(n="size", accuracy="mean")
        .sort_values(["model", "game", "rule_variant"])
        .reset_index(drop=True)
    )
    by_rule_path = results_dir / f"summary_by_game_rule__{args.run_tag}.csv"
    by_rule.to_csv(by_rule_path, index=False)

    runlog = pd.DataFrame(run_rows)
    runlog_path = results_dir / f"run_log__{args.run_tag}.csv"
    runlog.to_csv(runlog_path, index=False)

    print(f"[ok] combinations completed: {done}/{total}")
    print(f"[ok] predictions: {pred_path}")
    print(f"[ok] summary full: {summary_path}")
    print(f"[ok] summary game_rule: {by_rule_path}")
    print(f"[ok] run log: {runlog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
