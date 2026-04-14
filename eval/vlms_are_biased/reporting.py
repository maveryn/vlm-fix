from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROMPT_LABELS = {"original": "Original", "item_alias": "Alias"}
PROMPT_ORDER = {"Original": 0, "Alias": 1}
IMAGE_LABELS = {"original": "Original", "flipped": "Flipped"}
IMAGE_ORDER = {"Original": 0, "Flipped": 1}


def _prompt_label(style_key: str) -> str:
    return PROMPT_LABELS.get(str(style_key), str(style_key))


def _image_label(image_key: str) -> str:
    return IMAGE_LABELS.get(str(image_key), str(image_key))


def _build_metric_records(
    summary_df: pd.DataFrame,
    topics: list[str],
    metric_col: str,
) -> list[dict[str, object]]:
    include_image_variant = "image_variant" in summary_df.columns
    idx_cols = ["model", "prompt_style"]
    if include_image_variant:
        idx_cols = ["model", "image_variant", "prompt_style"]

    piv = summary_df.pivot_table(
        index=idx_cols,
        columns="topic",
        values=metric_col,
        aggfunc="mean",
    )
    for topic in topics:
        if topic not in piv.columns:
            piv[topic] = pd.NA
    piv = piv[topics]
    piv["Average_raw"] = piv.mean(axis=1)

    piv_df = piv.reset_index()
    group_cols = ["model", "image_variant"] if include_image_variant else ["model"]

    baseline_by_model: dict[str, float] = {}
    if include_image_variant:
        base_df = piv_df[
            (piv_df["image_variant"].astype(str).str.lower() == "original")
            & (piv_df["prompt_style"].astype(str).str.lower() == "original")
        ]
    else:
        base_df = piv_df[piv_df["prompt_style"].astype(str).str.lower() == "original"]
    for _, b in base_df.iterrows():
        base_avg = b.get("Average_raw", pd.NA)
        if pd.notna(base_avg):
            baseline_by_model[str(b["model"])] = float(base_avg)

    records: list[dict[str, object]] = []
    for group_key, grp in piv_df.groupby(group_cols, dropna=False):
        if include_image_variant:
            model_name, image_variant = group_key
            image_label = _image_label(str(image_variant))
        else:
            model_name = group_key
            image_label = None
            image_variant = None

        for style_key in ("original", "item_alias"):
            row_df = grp[grp["prompt_style"] == style_key]
            row = row_df.iloc[0] if not row_df.empty else None
            if row is None:
                rec = {
                    "Model": str(model_name),
                    "Prompt": _prompt_label(style_key),
                    "prompt_style": style_key,
                    "topic_vals": {t: None for t in topics},
                    "avg": None,
                    "avg_delta": None,
                }
                if image_label is not None:
                    rec["Image"] = image_label
                    rec["image_variant"] = str(image_variant)
                records.append(rec)
                continue

            topic_vals = {}
            for topic in topics:
                v = row.get(topic, pd.NA)
                topic_vals[topic] = None if pd.isna(v) else float(v)

            avg = row.get("Average_raw", pd.NA)
            avg = None if pd.isna(avg) else float(avg)

            avg_delta = None
            model_base = baseline_by_model.get(str(model_name))
            is_baseline_row = (style_key == "original") and (
                (not include_image_variant) or (str(image_variant).lower() == "original")
            )
            if (not is_baseline_row) and model_base is not None and avg is not None:
                avg_delta = avg - float(model_base)

            rec = {
                "Model": str(model_name),
                "Prompt": _prompt_label(style_key),
                "prompt_style": style_key,
                "topic_vals": topic_vals,
                "avg": avg,
                "avg_delta": avg_delta,
            }
            if image_label is not None:
                rec["Image"] = image_label
                rec["image_variant"] = str(image_variant)
            records.append(rec)

    def _sort_key(rec: dict[str, object]) -> tuple[object, ...]:
        model = str(rec["Model"])
        prompt_ord = PROMPT_ORDER.get(str(rec["Prompt"]), 99)
        if "Image" not in rec:
            return (model, prompt_ord)
        image_ord = IMAGE_ORDER.get(str(rec["Image"]), 99)
        return (model, image_ord, prompt_ord)

    records.sort(key=_sort_key)
    return records


def _format_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100.0:.1f}"


def _format_pct_delta(value: float | None) -> str:
    return "" if value is None else f"{value * 100.0:+.1f}"


def build_metric_display_table(
    summary_df: pd.DataFrame,
    topics: list[str],
    metric_col: str,
) -> pd.DataFrame:
    records = _build_metric_records(summary_df=summary_df, topics=topics, metric_col=metric_col)
    rows: list[dict[str, object]] = []
    include_image_variant = any("Image" in r for r in records)
    for rec in records:
        out: dict[str, object] = {"Model": rec["Model"]}
        if include_image_variant:
            out["Image"] = rec.get("Image", "")
        out["Prompt"] = rec["Prompt"]
        topic_vals = rec["topic_vals"]  # type: ignore[assignment]
        for topic in topics:
            out[topic] = _format_pct(topic_vals.get(topic))  # type: ignore[arg-type]
        avg = rec["avg"]  # type: ignore[assignment]
        avg_delta = rec["avg_delta"]  # type: ignore[assignment]
        if avg is None:
            out["Average"] = ""
        elif avg_delta is None:
            out["Average"] = _format_pct(avg)
        else:
            out["Average"] = f"{_format_pct(avg)} ({_format_pct_delta(avg_delta)})"
        rows.append(out)
    return pd.DataFrame(rows)


def build_combined_acc_bias_table(summary_df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    acc_records = _build_metric_records(summary_df=summary_df, topics=topics, metric_col="accuracy")
    bias_records = _build_metric_records(summary_df=summary_df, topics=topics, metric_col="bias_ratio")
    bias_index = {(r["Model"], r.get("Image", ""), r["Prompt"]): r for r in bias_records}

    rows: list[dict[str, object]] = []
    include_image_variant = any("Image" in r for r in acc_records)
    for acc in acc_records:
        key = (acc["Model"], acc.get("Image", ""), acc["Prompt"])
        bias = bias_index.get(key)
        if bias is None:
            continue

        out: dict[str, object] = {"Model": acc["Model"]}
        if include_image_variant:
            out["Image"] = acc.get("Image", "")
        out["Prompt"] = acc["Prompt"]
        acc_topic_vals = acc["topic_vals"]  # type: ignore[assignment]
        bias_topic_vals = bias["topic_vals"]  # type: ignore[assignment]
        for topic in topics:
            a = acc_topic_vals.get(topic)  # type: ignore[arg-type]
            b = bias_topic_vals.get(topic)  # type: ignore[arg-type]
            if a is None or b is None:
                out[topic] = ""
            else:
                out[topic] = f"{_format_pct(a)}/{_format_pct(b)}"

        a_avg = acc["avg"]  # type: ignore[assignment]
        b_avg = bias["avg"]  # type: ignore[assignment]
        a_delta = acc["avg_delta"]  # type: ignore[assignment]
        b_delta = bias["avg_delta"]  # type: ignore[assignment]
        if a_avg is None or b_avg is None:
            out["Average"] = ""
        elif a_delta is None or b_delta is None:
            out["Average"] = f"{_format_pct(a_avg)} / {_format_pct(b_avg)}"
        else:
            out["Average"] = (
                f"{_format_pct(a_avg)} ({_format_pct_delta(a_delta)}) / "
                f"{_format_pct(b_avg)} ({_format_pct_delta(b_delta)})"
            )
        rows.append(out)

    return pd.DataFrame(rows)


def _latex_escape(text: str) -> str:
    s = str(text)
    s = s.replace("\\", "\\textbackslash{}")
    for ch, repl in (
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
    ):
        s = s.replace(ch, repl)
    return s


def dataframe_to_booktabs_latex(
    df: pd.DataFrame,
    topics: list[str],
    caption: str = "VLMs-Are-Biased accuracy/bias summary (percent).",
    label: str = "tab:vlms_are_biased_acc_bias",
) -> str:
    include_image_variant = "Image" in df.columns
    headers = ["Model", "Prompt", *topics, "Average"]
    col_spec_prefix = "ll"
    if include_image_variant:
        headers = ["Model", "Image", "Prompt", *topics, "Average"]
        col_spec_prefix = "lll"
    col_spec = col_spec_prefix + "c" * (len(topics) + 1)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(_latex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        vals = [row.get(h, "") for h in headers]
        lines.append(" & ".join(_latex_escape(str(v)) for v in vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def write_matrix_reports(
    *,
    summary_df: pd.DataFrame,
    topics: Iterable[str],
    out_dir: str | Path,
    summary_matrix_csv_name: str,
    bias_matrix_csv_name: str,
    summary_xlsx_name: str,
    combined_latex_path: str | Path | None = None,
    details_csv_name: str | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    topics_list = list(topics)

    summary_display = build_metric_display_table(summary_df=summary_df, topics=topics_list, metric_col="accuracy")
    bias_display = build_metric_display_table(summary_df=summary_df, topics=topics_list, metric_col="bias_ratio")
    combined_display = build_combined_acc_bias_table(summary_df=summary_df, topics=topics_list)

    summary_matrix_csv = out_path / summary_matrix_csv_name
    bias_matrix_csv = out_path / bias_matrix_csv_name
    xlsx_path = out_path / summary_xlsx_name

    summary_display.to_csv(summary_matrix_csv, index=False)
    bias_display.to_csv(bias_matrix_csv, index=False)

    with pd.ExcelWriter(xlsx_path) as writer:
        summary_display.to_excel(writer, sheet_name="summary", index=False)
        bias_display.to_excel(writer, sheet_name="bias_summary", index=False)
        combined_display.to_excel(writer, sheet_name="acc_bias", index=False)
        summary_df.to_excel(writer, sheet_name="details", index=False)

    out = {
        "summary_matrix_csv": str(summary_matrix_csv),
        "bias_matrix_csv": str(bias_matrix_csv),
        "summary_xlsx": str(xlsx_path),
    }

    if details_csv_name:
        details_csv_path = out_path / details_csv_name
        summary_df.to_csv(details_csv_path, index=False)
        out["details_csv"] = str(details_csv_path)

    if combined_latex_path is not None:
        latex_path = Path(combined_latex_path)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_text = dataframe_to_booktabs_latex(df=combined_display, topics=topics_list)
        latex_path.write_text(latex_text, encoding="utf-8")
        out["latex"] = str(latex_path)

    return out
