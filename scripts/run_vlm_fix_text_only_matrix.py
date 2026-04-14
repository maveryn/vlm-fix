#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from eval.model_registry import MODEL_REGISTRY  # noqa: E402
from vlm_fix.dataset_io import load_instances_frame  # noqa: E402

OPEN_SOURCE_10_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-14B",
    "allenai/Molmo2-4B",
    "allenai/Molmo2-8B",
]

API_4_MODELS = [
    "gpt-4.1",
    "gpt-5.2",
    "claude-sonnet-4-0",
    "claude-sonnet-4-5",
]

ALL_14_MODELS = OPEN_SOURCE_10_MODELS + API_4_MODELS
GAMES = ["tictactoe", "reversi", "connect4", "dots_boxes"]

SUPPRESSED_STDERR_PATTERNS = [
    "oneDNN custom operations are on",
    "Unable to register cuFFT factory",
    "Unable to register cuDNN factory",
    "Unable to register cuBLAS factory",
    "This TensorFlow binary is optimized to use available CPU instructions",
    "All log messages before absl::InitializeLog() is called are written to STDERR",
    "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
    "Qwen2VLImageProcessor",
    "is now loaded as a fast processor by default",
    "The argument `trust_remote_code` is to be used with Auto classes.",
    "Failed to JIT torch c dlpack extension",
    "pip install torch-c-dlpack-ext",
    "I tensorflow/core/",
    "E external/local_xla/",
    "E0000 00:00:",
]


class _FilteredStderr:
    def __init__(self, wrapped, suppress_patterns: List[str]) -> None:
        self._wrapped = wrapped
        self._patterns = [str(x) for x in suppress_patterns]
        self._buf = ""

    def _suppress(self, line: str) -> bool:
        line_s = str(line)
        if line_s.startswith("INFO ") and "[vllm" in line_s:
            return True
        return any(p in line_s for p in self._patterns)

    def write(self, data):
        s = str(data)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not self._suppress(line):
                self._wrapped.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            if not self._suppress(self._buf):
                self._wrapped.write(self._buf)
            self._buf = ""
        self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()


def _configure_quiet_mode(quiet: bool) -> None:
    if not quiet:
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    warnings.filterwarnings("ignore")
    for name in ["absl", "tensorflow", "transformers", "vllm", "torch.distributed", "urllib3.connectionpool"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr, SUPPRESSED_STDERR_PATTERNS)


@contextmanager
def _suppress_stderr_fd(enabled: bool):
    if not enabled:
        yield
        return
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


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
    matches: List[Tuple[int, str]] = []
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


def _build_message(row: pd.Series) -> List[Dict[str, str]]:
    text = str(row.get("input_text", "")).strip()
    if not text:
        prompt = str(row.get("prompt", "")).strip()
        board = str(row.get("board_text", "")).strip()
        text = prompt if not board else (prompt + "\n\nBoard:\n" + board)
    return [{"type": "text", "value": text}]


def _run_single_with_model(
    model,
    model_name: str,
    game: str,
    instances: pd.DataFrame,
    out_jsonl: Path,
    batch_size: int,
    combo_i: int,
    combo_total: int,
) -> pd.DataFrame:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, object]] = []

    with out_jsonl.open("w", encoding="utf-8") as f:
        desc = f"[{combo_i}/{combo_total}] {game} | {model_name}"
        for start in tqdm(range(0, len(instances), batch_size), desc=desc, unit="batch", leave=False):
            chunk = instances.iloc[start : start + batch_size]
            messages = [_build_message(row) for _, row in chunk.iterrows()]
            preds, metas = model.generate_batch_with_meta(messages, dataset="vlm_fix_text_only")

            for (_, src), pred, meta in zip(chunk.iterrows(), preds, metas):
                labels = _split_labels(str(src["valid_labels"]))
                extracted = _extract_answer(str(pred), labels)
                answer = str(src["answer"])
                rec = {
                    "model": model_name,
                    "game": str(src["game"]),
                    "index": int(src["index"]),
                    "state_id": int(src["state_id"]),
                    "render_variant": str(src.get("render_variant", "canonical_text")),
                    "rule_variant": str(src["rule_variant"]),
                    "prompt_variant": str(src.get("prompt_variant", "standard")),
                    "prompt_type": str(src.get("prompt_type", "direct")),
                    "question_target": str(src["question_target"]),
                    "image_text_order": str(src.get("image_text_order", "text_first")),
                    "prompt": str(src.get("prompt", "")),
                    "board_text": str(src.get("board_text", "")),
                    "input_text": str(src.get("input_text", "")),
                    "valid_labels": str(src["valid_labels"]),
                    "answer": answer,
                    "prediction": str(pred),
                    "extracted_answer": extracted,
                    "correct": bool(extracted == answer),
                    "output_tokens": (meta or {}).get("output_tokens") if isinstance(meta, dict) else None,
                    "finish_reason": (meta or {}).get("finish_reason") if isinstance(meta, dict) else None,
                }
                rows_out.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows_out)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "game", "rule_variant", "prompt_variant", "prompt_type", "question_target"]
    return (
        df.groupby(keys, dropna=False)["correct"]
        .agg(n="size", accuracy="mean")
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM-Fix text-only matrix (model x game).")
    parser.add_argument("--dataset-dir", type=str, default="data/generated/vlm_fix_text_only")
    parser.add_argument("--dataset-source", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-config", type=str, default="vlm_fix_text_only")
    parser.add_argument("--hf-split", type=str, default="main")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-cache-dir", type=str, default="data/hf_cache")
    parser.add_argument("--models", nargs="+", default=ALL_14_MODELS)
    parser.add_argument("--games", nargs="+", default=GAMES)
    parser.add_argument("--rule-variants", nargs="+", choices=["standard", "inverse"], default=None)
    parser.add_argument("--question-targets", nargs="+", choices=["winner", "loser"], default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--allow-hf-fallback", action="store_true", default=False)
    parser.add_argument("--runs-dir", type=str, default="runs/vlm_fix_text_only")
    parser.add_argument("--results-dir", type=str, default="results/vlm_fix_text_only")
    parser.add_argument("--run-tag", type=str, default="prompt-direct")
    parser.add_argument("--max-rows-per-game", type=int, default=0)
    parser.add_argument("--quiet", dest="quiet", action="store_true", default=True)
    parser.add_argument("--no-quiet", dest="quiet", action="store_false")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--continue-on-init-error", dest="continue_on_init_error", action="store_true", default=True)
    parser.add_argument("--no-continue-on-init-error", dest="continue_on_init_error", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_quiet_mode(bool(args.quiet))

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")

    df, _dataset_dir = load_instances_frame(
        dataset_source=args.dataset_source,
        dataset_dir=args.dataset_dir,
        hf_repo=args.hf_repo,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
        hf_cache_dir=args.hf_cache_dir,
        require_image_column=False,
    )

    df = df[df["game"].isin(args.games)].copy()
    if args.rule_variants:
        df = df[df["rule_variant"].isin(args.rule_variants)].copy()
    if args.question_targets:
        df = df[df["question_target"].isin(args.question_targets)].copy()
    if args.max_rows_per_game and int(args.max_rows_per_game) > 0:
        k = int(args.max_rows_per_game)
        df = (
            df.sort_values(["game", "state_id", "rule_variant", "question_target", "index"])
            .groupby("game", as_index=False, group_keys=False)
            .head(k)
            .reset_index(drop=True)
        )
    if df.empty:
        raise ValueError("No rows left after filtering.")

    print(f"[filter] rows after filters: total={len(df)} | per_game={df.groupby('game').size().to_dict()}")

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

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

    combo_bar = tqdm(total=total, desc="combinations", unit="combo")
    i = 0
    for model_name in args.models:
        model_games = [g for g in args.games if not df[df["game"] == g].empty]
        if not model_games:
            continue

        ctor = MODEL_REGISTRY[model_name]
        try:
            ctor_kwargs: Dict[str, object] = {
                "vllm_batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
            }
            if args.gpu_memory_utilization is not None:
                ctor_kwargs["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
            if args.max_model_len is not None:
                ctor_kwargs["max_model_len"] = int(args.max_model_len)
            if args.max_num_batched_tokens is not None:
                ctor_kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
            if bool(args.allow_hf_fallback):
                ctor_kwargs["require_vllm"] = False
            with _suppress_stderr_fd(enabled=bool(args.quiet)):
                model = ctor(**ctor_kwargs)
        except Exception as exc:
            if not bool(args.continue_on_init_error):
                raise
            err_text = f"{type(exc).__name__}: {exc}"
            print(f"[error] init failed for model={model_name}: {err_text}")
            for game in model_games:
                i += 1
                combo_bar.set_postfix_str(f"{model_name} | {game}")
                run_rows.append(
                    {
                        "model": model_name,
                        "game": game,
                        "status": "init_failed",
                        "error": err_text,
                        "n": int(len(df[df["game"] == game])),
                    }
                )
                done += 1
                combo_bar.update(1)
            continue
        try:
            for game in model_games:
                i += 1
                game_df = df[df["game"] == game].copy()
                out_jsonl = runs_dir / game / f"{_safe_model_name(model_name)}__{args.run_tag}.jsonl"
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
                    print(f"[warn] existing file length mismatch ({len(loaded)} != {len(game_df)}), rerunning: {out_jsonl}")

                pred_df = _run_single_with_model(
                    model=model,
                    model_name=model_name,
                    game=game,
                    instances=game_df,
                    out_jsonl=out_jsonl,
                    batch_size=args.batch_size,
                    combo_i=i,
                    combo_total=total,
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
        finally:
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    combo_bar.close()

    runlog = pd.DataFrame(run_rows)
    runlog_path = results_dir / f"run_log__{args.run_tag}.csv"
    runlog.to_csv(runlog_path, index=False)

    pred_path = results_dir / f"predictions_all__{args.run_tag}.parquet"
    summary_path = results_dir / f"summary_full_combo__{args.run_tag}.csv"
    by_rule_path = results_dir / f"summary_by_game_rule__{args.run_tag}.csv"

    if not all_frames:
        pd.DataFrame(
            columns=[
                "model",
                "game",
                "rule_variant",
                "prompt_variant",
                "prompt_type",
                "question_target",
                "n",
                "accuracy",
            ]
        ).to_csv(summary_path, index=False)
        pd.DataFrame(columns=["model", "game", "rule_variant", "n", "accuracy"]).to_csv(by_rule_path, index=False)
        print(f"[warn] no prediction rows were produced; all runs failed or were skipped with no data.")
        print(f"[ok] run log: {runlog_path}")
        return 0

    pred_all = pd.concat(all_frames, ignore_index=True)
    pred_all.to_parquet(pred_path, index=False)

    summary = _summarize(pred_all)
    summary.to_csv(summary_path, index=False)

    by_rule = (
        pred_all.groupby(["model", "game", "rule_variant"], as_index=False)["correct"]
        .agg(n="size", accuracy="mean")
        .sort_values(["model", "game", "rule_variant"])
        .reset_index(drop=True)
    )
    by_rule.to_csv(by_rule_path, index=False)

    print(f"[ok] combinations completed: {done}/{total}")
    print(f"[ok] predictions: {pred_path}")
    print(f"[ok] summary full: {summary_path}")
    print(f"[ok] summary game_rule: {by_rule_path}")
    print(f"[ok] run log: {runlog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
