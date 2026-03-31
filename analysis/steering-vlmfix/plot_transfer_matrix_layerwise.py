#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def _load_runner_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_transfer_matrix_cached", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_transfer_matrix_cached"] = mod
    spec.loader.exec_module(mod)
    return mod


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _pretty_model_name(name: str) -> str:
    n = str(name).strip()
    if "/" in n:
        n = n.split("/", 1)[1]
    n = n.replace("-Instruct", "")
    return n


def _pretty_game_name(game: str) -> str:
    g = str(game)
    mapping = {
        "tictactoe": "Tic-Tac-Toe",
        "reversi": "Reversi",
        "connect4": "Connect4",
        "dots_boxes": "Dots and Boxes",
    }
    return mapping.get(g, g)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-setup layerwise plots for transfer-matrix runs.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "analysis/steering-vlmfix/outputs/transfer_matrix_cached/"
            "qwen25vl7b_transfer_matrix_spg100_rep3_last12_alpha1_full6_restart"
        ),
    )
    p.add_argument("--dataset", type=Path, default=Path("data/generated/vlm_fix/instances.parquet"))
    p.add_argument("--dataset-root", type=Path, default=Path("data/generated/vlm_fix"))
    p.add_argument(
        "--runner-script",
        type=Path,
        default=Path("analysis/steering-vlmfix/run_transfer_matrix_cached.py"),
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--skip-source-recompute", action="store_true")
    return p.parse_args()


def compute_source_accuracy_by_setup(
    *,
    run_dir: Path,
    dataset: pd.DataFrame,
    dataset_root: Path,
    runner_mod,
    max_new_tokens: int,
) -> pd.DataFrame:
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    split_manifest = pd.read_csv(run_dir / "split_manifest.csv")
    runs = pd.read_csv(run_dir / "summary_runs.csv")

    model_name = str(run_meta["model"])
    prompt_type = str(run_meta["prompt_type"])
    render_variant = str(run_meta["render_variant"])
    prompt_variant = str(run_meta["prompt_variant"])

    family = runner_mod.detect_family(model_name)
    processor = runner_mod.make_processor(model_name, family)
    model = runner_mod.load_model(model_name)
    if family == "internvl":
        try:
            image_tok = getattr(processor, "image_token", "<IMG_CONTEXT>")
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                model.img_context_token_id = int(tok.convert_tokens_to_ids(image_tok))
        except Exception:
            pass

    setups = (
        runs[["setup_id", "setup_style", "source_game"]]
        .drop_duplicates()
        .sort_values(["setup_id"])
        .reset_index(drop=True)
    )

    # Cache game/style rows so we only build prompts once per unique source combo.
    source_rows_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    out_rows: List[dict] = []

    needed = {(str(r.source_game), str(r.setup_style)) for r in setups.itertuples(index=False)}
    for game, style in sorted(needed):
        source_rows_cache[(game, style)] = runner_mod.load_game_rows(
            dataset=dataset,
            game=game,
            prompt_type=prompt_type,
            render_variant=render_variant,
            prompt_variant=prompt_variant,
            selected_states=sorted(dataset.loc[
                (dataset["game"] == game)
                & (dataset["prompt_type"] == prompt_type)
                & (dataset["render_variant"] == render_variant)
                & (dataset["prompt_variant"] == prompt_variant),
                "state_id",
            ].unique().tolist()),
            style=style,
        )

    pbar = tqdm(
        setups.itertuples(index=False),
        total=len(setups),
        desc="source_acc_setups",
        unit="setup",
    )
    for rec in pbar:
        setup_id = str(rec.setup_id)
        setup_style = str(rec.setup_style)
        source_game = str(rec.source_game)
        src_rows = source_rows_cache[(source_game, setup_style)]

        for repeat in sorted(runs.loc[runs["setup_id"] == setup_id, "repeat"].unique().tolist()):
            repeat = int(repeat)
            test_states = set(
                int(x)
                for x in split_manifest.loc[
                    (split_manifest["game"] == source_game)
                    & (split_manifest["repeat"] == repeat)
                    & (split_manifest["split"] == "test"),
                    "state_id",
                ].tolist()
            )
            src_test = src_rows[src_rows["state_id"].isin(test_states)].copy().reset_index(drop=True)
            if src_test.empty:
                raise RuntimeError(f"No source test rows for setup={setup_id}, repeat={repeat}")

            correct = 0
            for r in src_test.itertuples(index=False):
                image_abs = str((dataset_root / str(r.image_path)).resolve())
                inp = runner_mod.build_inputs(
                    processor=processor,
                    family=family,
                    prompt=str(r.prompt),
                    image_abs_path=image_abs,
                    image_text_order=str(r.image_text_order),
                )
                pred = runner_mod.generate_plain(
                    model=model,
                    processor=processor,
                    inputs_cpu=inp,
                    max_new_tokens=int(max_new_tokens),
                )
                labels = runner_mod.split_labels(str(r.valid_labels))
                ex = runner_mod.extract_answer(pred, labels)
                correct += int(ex == str(r.answer))

            n = int(len(src_test))
            out_rows.append(
                {
                    "setup_id": setup_id,
                    "repeat": repeat,
                    "source_game": source_game,
                    "setup_style": setup_style,
                    "source_acc": float(correct / n),
                    "n": n,
                }
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(out_rows).sort_values(["setup_id", "repeat"]).reset_index(drop=True)
    df.to_csv(run_dir / "source_acc_by_setup_repeat.csv", index=False)

    agg = (
        df.groupby(["setup_id", "source_game", "setup_style"], as_index=False)
        .agg(
            source_acc_mean=("source_acc", "mean"),
            source_acc_std=("source_acc", "std"),
            n_total=("n", "sum"),
            repeats=("repeat", "nunique"),
        )
        .sort_values(["setup_id"])
        .reset_index(drop=True)
    )
    agg.to_csv(run_dir / "source_acc_by_setup.csv", index=False)
    return agg


def plot_all_setups(run_dir: Path, source_df: pd.DataFrame, dpi: int = 220) -> None:
    layer_df = pd.read_csv(run_dir / "summary_layer_setup_meanstd.csv")
    runs_df = pd.read_csv(run_dir / "summary_runs.csv")
    run_meta_path = run_dir / "run_meta.json"
    model_name = "Model"
    if run_meta_path.exists():
        try:
            model_name = _pretty_model_name(json.loads(run_meta_path.read_text(encoding="utf-8")).get("model", "Model"))
        except Exception:
            model_name = "Model"
    plots_dir = run_dir / "plots_layerwise"
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs_for_std = runs_df.rename(
        columns={
            "routing_rule_acc_target": "routing_rule_acc",
            "routing_answer_acc_target": "routing_ans_acc",
        }
    )
    routing_std_df = (
        runs_for_std.groupby(["setup_id", "layer_paper"], as_index=False)
        .agg(
            routing_rule_std=("routing_rule_acc", "std"),
            routing_ans_std=("routing_ans_acc", "std"),
        )
        .fillna(0.0)
    )

    setup_ids = sorted(layer_df["setup_id"].unique().tolist())
    for sid in setup_ids:
        d = layer_df[layer_df["setup_id"] == sid].copy().sort_values("layer_paper")
        srow = source_df[source_df["setup_id"] == sid]
        if srow.empty:
            raise RuntimeError(f"No source_acc row for setup_id={sid}")
        source_mean = float(srow["source_acc_mean"].iloc[0])
        source_std = float(np.nan_to_num(srow["source_acc_std"].iloc[0], nan=0.0))

        x = d["layer_paper"].to_numpy()
        base = d["base_acc_mean"].to_numpy()
        base_std = np.nan_to_num(d["base_acc_std"].to_numpy(), nan=0.0)
        patch = d["patched_acc_mean"].to_numpy()
        patch_std = np.nan_to_num(d["patched_acc_std"].to_numpy(), nan=0.0)
        rr = d["routing_rule_mean"].to_numpy()
        ra = d["routing_ans_mean"].to_numpy()
        rstd = (
            routing_std_df[routing_std_df["setup_id"] == sid]
            .set_index("layer_paper")
            .reindex(d["layer_paper"])
            .fillna(0.0)
        )
        rr_std = rstd["routing_rule_std"].to_numpy()
        ra_std = rstd["routing_ans_std"].to_numpy()
        src_game = str(d["source_game"].iloc[0])
        tgt_game = str(d["target_game"].iloc[0])
        style = str(d["setup_style"].iloc[0])

        fig, ax = plt.subplots(1, 1, figsize=(4.8, 2.88), constrained_layout=True)
        is_within = src_game == tgt_game
        base_label = "Base" if is_within else "Target (Base)"
        ax.plot(x, base, marker="o", lw=2.0, color="#c0392b", label=base_label)
        ax.fill_between(x, base - base_std, base + base_std, color="#c0392b", alpha=0.15, linewidth=0)
        ax.plot(x, patch, marker="s", lw=2.0, color="#1f77b4", label="Patched")
        ax.fill_between(x, patch - patch_std, patch + patch_std, color="#1f77b4", alpha=0.15, linewidth=0)
        if (not is_within) and (not np.isnan(source_mean)):
            ax.plot(
                x,
                np.full_like(x, source_mean, dtype=float),
                marker="P",
                lw=2.0,
                ls="--",
                color="#2e7d32",
                label="Source",
            )
            if source_std > 0:
                ax.fill_between(
                    x,
                    np.full_like(x, source_mean - source_std, dtype=float),
                    np.full_like(x, source_mean + source_std, dtype=float),
                    color="#2e7d32",
                    alpha=0.12,
                    linewidth=0,
                )
        ax.plot(x, rr, marker="^", lw=2.0, color="#8e44ad", alpha=0.95, label="Routing Rule")
        ax.fill_between(x, rr - rr_std, rr + rr_std, color="#8e44ad", alpha=0.12, linewidth=0)
        ax.plot(x, ra, marker="D", lw=2.0, color="#f39c12", alpha=0.95, label="Routing Answer")
        ax.fill_between(x, ra - ra_std, ra + ra_std, color="#f39c12", alpha=0.12, linewidth=0)
        ymin_candidates = [
            float(np.min(base - base_std)),
            float(np.min(patch - patch_std)),
            float(np.min(rr - rr_std)),
            float(np.min(ra - ra_std)),
        ]
        ymax_candidates = [
            float(np.max(base + base_std)),
            float(np.max(patch + patch_std)),
            float(np.max(rr + rr_std)),
            float(np.max(ra + ra_std)),
        ]
        if not is_within:
            ymin_candidates.append(float(source_mean - source_std))
            ymax_candidates.append(float(source_mean + source_std))

        ymin = max(0.0, min(ymin_candidates) - 0.03)
        ymax = min(1.05, max(ymax_candidates) + 0.03)
        ymax = max(1.02, ymax)
        if ymax - ymin < 0.20:
            mid = 0.5 * (ymin + ymax)
            ymin = max(0.0, mid - 0.10)
            ymax = min(1.05, mid + 0.10)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title("")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=True, ncol=1)

        if src_game == tgt_game:
            fig.suptitle(
                f"{model_name}: {_pretty_game_name(src_game)}",
                fontsize=11,
                y=1.035,
            )
        else:
            fig.suptitle(f"{sid} | {src_game} -> {tgt_game} | style={style}", fontsize=11, y=1.035)

        stem = _safe(sid)
        png = plots_dir / f"{stem}.png"
        pdf = plots_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
        fig.savefig(pdf, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)

    # Optional combined summary panel.
    n = len(setup_ids)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, sid in enumerate(setup_ids):
        ax = axes[i]
        d = layer_df[layer_df["setup_id"] == sid].copy().sort_values("layer_paper")
        srow = source_df[source_df["setup_id"] == sid]
        source_mean = float(srow["source_acc_mean"].iloc[0])
        x = d["layer_paper"].to_numpy()
        ax.plot(x, d["base_acc_mean"].to_numpy(), lw=1.7, color="#c0392b", label="Target")
        ax.plot(x, d["patched_acc_mean"].to_numpy(), lw=1.7, color="#1f77b4", label="Patched")
        if not np.isnan(source_mean):
            ax.plot(x, np.full_like(x, source_mean, dtype=float), lw=1.7, ls="--", color="#2e7d32", label="Source")
        ax.plot(x, d["routing_rule_mean"].to_numpy(), lw=1.7, color="#8e44ad", alpha=0.9, label="RouteRule")
        ax.plot(x, d["routing_ans_mean"].to_numpy(), lw=1.7, color="#f39c12", alpha=0.9, label="RouteAns")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(sid, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Acc")
        ax.grid(alpha=0.2)

    for j in range(len(setup_ids), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True)
    fig.suptitle("Transfer Matrix Layerwise Overview", fontsize=13, y=1.01)
    fig.savefig(plots_dir / "all_setups_overview.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(plots_dir / "all_setups_overview.pdf", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    runner_mod = _load_runner_module(args.runner_script.resolve())

    source_summary_path = run_dir / "source_acc_by_setup.csv"
    if source_summary_path.exists():
        source_df = pd.read_csv(source_summary_path)
    elif args.skip_source_recompute:
        # Lightweight fallback for plotting-only workflows where source bands are not required.
        layer_df = pd.read_csv(run_dir / "summary_layer_setup_meanstd.csv")
        source_df = (
            layer_df[["setup_id", "source_game", "setup_style"]]
            .drop_duplicates()
            .assign(source_acc_mean=np.nan, source_acc_std=0.0, n_total=0, repeats=0)
            .reset_index(drop=True)
        )
    else:
        dataset = pd.read_parquet(args.dataset)
        source_df = compute_source_accuracy_by_setup(
            run_dir=run_dir,
            dataset=dataset,
            dataset_root=args.dataset_root.resolve(),
            runner_mod=runner_mod,
            max_new_tokens=int(args.max_new_tokens),
        )

    plot_all_setups(run_dir=run_dir, source_df=source_df, dpi=int(args.dpi))
    print("[ok] plots written:", run_dir / "plots_layerwise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
