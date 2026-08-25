"""Generate normative deviation reports and figures."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fia_ml.normative.config import NormativeConfig
from fia_ml.paths import PROJECT_ROOT, TARGET_MAPPING_CONFIG, ensure_dir
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.evaluate import plot_confusion_matrix
from fia_ml.utils import secure_file_io as sio


def _label_names() -> dict[str, str]:
    mapping = load_target_mapping(TARGET_MAPPING_CONFIG)
    return {str(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _write_breakdown_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    pd.DataFrame(records).to_csv(output_path, index=False)


def _plot_deviation_by_incident_type(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    if not records:
        return

    frame = pd.DataFrame(records)
    if "incident_type" not in frame.columns or frame.empty:
        return

    frame = frame.sort_values("disagreement_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(frame))))
    sns.barplot(
        data=frame,
        y="incident_type",
        x="disagreement_rate",
        hue="incident_type",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Disagreement rate (FIA vs normative)")
    ax.set_ylabel("Incident type")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _write_markdown_summary(
    comparison: dict[str, Any],
    cfg: NormativeConfig,
    output_path: Path,
    *,
    assumptions: list[str] | None = None,
    rules_version: str | None = None,
) -> None:
    agg = comparison["aggregate"]
    lines = [
        f"# Normative Deviation Report — {date.today().isoformat()}",
        "",
        "## Interpretation",
        "",
        "Normative outcomes reflect one documented, rule-based interpretation of racing",
        "regulations. They are **not** ground truth and are intended for research comparison",
        "against FIA stewarding decisions.",
        "",
    ]

    if rules_version:
        lines.extend([f"- Rules version: `{rules_version}`", ""])

    if assumptions:
        lines.extend(["## Rule assumptions", ""])
        lines.extend(f"- {item}" for item in assumptions)
        lines.append("")

    lines.extend(
        [
            "## Aggregate metrics (FIA vs normative)",
            "",
            f"- Rows compared: **{agg['row_count']}**",
            f"- Agreement rate: **{_format_pct(agg['agreement_rate'])}**",
            f"- Cohen's kappa: **{agg['cohens_kappa']:.3f}**",
            f"- Mean deviation direction (normative − FIA): **{agg['mean_deviation_direction']:.3f}**",
            f"- FIA harsher rate: **{_format_pct(agg['fia_harsher_rate'])}**",
            f"- Normative harsher rate: **{_format_pct(agg['normative_harsher_rate'])}**",
            f"- `manual_review` rate: **{_format_pct(agg['manual_review_rate'])}**",
            "",
        ]
    )

    if agg.get("matched_row_count"):
        lines.extend(
            [
                "### Excluding `manual_review` rows",
                "",
                f"- Matched rows: **{agg['matched_row_count']}**",
                f"- Agreement rate: **{_format_pct(agg['agreement_rate_excl_manual_review'])}**",
                f"- Cohen's kappa: **{agg['cohens_kappa_excl_manual_review']:.3f}**",
                "",
            ]
        )

    ml = comparison.get("ml_comparison")
    if ml and ml.get("overlap_rows", 0) > 0:
        lines.extend(
            [
                "## Optional ML comparison (validation overlap only)",
                "",
                f"- Overlap rows: **{ml['overlap_rows']}**",
                f"- FIA vs ML agreement: **{_format_pct(ml['agreement_fia_ml'])}**",
                f"- Normative vs ML agreement: **{_format_pct(ml['agreement_normative_ml'])}**",
                f"- FIA vs normative agreement (overlap): **{_format_pct(ml['agreement_fia_normative'])}**",
                "",
            ]
        )

    breakdowns = comparison.get("breakdowns", {})
    incident_rows = breakdowns.get("incident_type", [])
    if incident_rows:
        lines.extend(["## Highest disagreement by incident type", ""])
        lines.append("| Incident type | n | Disagreement | FIA harsher | Normative harsher |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in incident_rows[:10]:
            lines.append(
                f"| {row.get('incident_type', '')} | {row.get('n', 0)} | "
                f"{_format_pct(row.get('disagreement_rate'))} | "
                f"{_format_pct(row.get('fia_harsher_rate'))} | "
                f"{_format_pct(row.get('normative_harsher_rate'))} |"
            )
        lines.append("")

    top_deviations = comparison.get("top_deviations", [])
    if top_deviations:
        lines.extend(["## Top deviation cases", ""])
        for idx, row in enumerate(top_deviations[:10], start=1):
            lines.append(
                f"{idx}. `{row.get('row_id', row.get('incident_id', ''))}` — "
                f"FIA severity {row.get('penalty_severity')} vs normative "
                f"{row.get('normative_penalty_severity')} "
                f"(`{row.get('normative_rule_id', '')}`)"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_deviation_report(
    comparison: dict[str, Any],
    cfg: NormativeConfig,
    report_dir: Path,
    *,
    assumptions: list[str] | None = None,
    rules_version: str | None = None,
) -> dict[str, str]:
    """Write markdown summary, CSV breakdowns, and figures."""
    report_dir = ensure_dir(report_dir)
    figures_dir = ensure_dir(report_dir / "figures")
    stamp = date.today().isoformat()
    label_names = _label_names()
    outputs: dict[str, str] = {}

    summary_path = report_dir / f"deviation_summary_{stamp}.md"
    _write_markdown_summary(
        comparison,
        cfg,
        summary_path,
        assumptions=assumptions,
        rules_version=rules_version,
    )
    outputs["summary_md"] = str(summary_path.relative_to(PROJECT_ROOT))

    breakdown_files = {
        "incident_type": "deviation_by_incident_type.csv",
        "session": "deviation_by_session.csv",
        "circuit": "deviation_by_circuit.csv",
        "season": "deviation_by_season.csv",
    }
    for dim, filename in breakdown_files.items():
        records = comparison.get("breakdowns", {}).get(dim, [])
        if not records:
            continue
        csv_path = report_dir / filename
        _write_breakdown_csv(records, csv_path)
        outputs[f"breakdown_{dim}"] = str(csv_path.relative_to(PROJECT_ROOT))

    confusion = comparison["aggregate"]["confusion_matrix"]
    confusion_path = figures_dir / "fia_vs_normative_confusion.png"
    plot_confusion_matrix(
        confusion["matrix"],
        confusion["labels"],
        label_names,
        confusion_path,
        title="FIA actual vs normative severity",
    )
    outputs["confusion_figure"] = str(confusion_path.relative_to(PROJECT_ROOT))

    incident_records = comparison.get("breakdowns", {}).get("incident_type", [])
    deviation_chart_path = figures_dir / "deviation_rate_by_incident_type.png"
    _plot_deviation_by_incident_type(
        incident_records,
        deviation_chart_path,
        title="Disagreement rate by incident type",
    )
    outputs["deviation_by_type_figure"] = str(deviation_chart_path.relative_to(PROJECT_ROOT))

    return outputs


def write_evaluation_metrics(
    comparison: dict[str, Any],
    models_dir: Path,
) -> Path:
    """Persist aggregate comparison metrics for downstream tooling."""
    models_dir = ensure_dir(models_dir)
    payload = {
        "generated_at": date.today().isoformat(),
        "aggregate": comparison["aggregate"],
    }
    if "ml_comparison" in comparison:
        payload["ml_comparison"] = comparison["ml_comparison"]
    output_path = models_dir / "evaluation_metrics.json"
    sio.write_json(output_path, payload)
    return output_path
