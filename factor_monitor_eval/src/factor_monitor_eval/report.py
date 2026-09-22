"""Markdown report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .io import write_text


def _fmt(value: Any) -> str:
    if value is None: return "N/A"
    if isinstance(value, float): return f"{value:.6f}"
    return str(value)


def _overall(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("slices", {}).get("overall")
    if isinstance(overall, dict):
        return overall
    return {}


def write_report(summary: dict[str, Any], path) -> None:
    lines = [
        "# Factor Monitor Evaluation Report", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Metric adapter: `{summary.get('metric_adapter', 'unknown')}`", "",
        "## Overall", "",
        "| rows | applied | not-applied | TPR | TNR | G-Mean² | invalid |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = _overall(summary)
    if not overall:
        lines += ["", "## Track summary"]
        for name in ("factor_level_human_semantic", "trajectory_any_path_human_semantic"):
            groups = summary.get(name, {})
            if groups:
                lines += ["", f"### {name}", "", "| factor level | rows | TPR | TNR | G-Mean² | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|---:|"]
                for key, row in groups.items():
                    lines.append(f"| {key} | {row.get('rows', row.get('trajectories_by_level', 'N/A'))} | {_fmt(row.get('tpr'))} | {_fmt(row.get('tnr'))} | {_fmt(row.get('gmean2', row.get('gmean2_with_tnr_convention')))} | {_fmt(row.get('auroc'))} | {_fmt(row.get('auprc'))} |")
        if summary.get("by_source_and_factor_level"):
            lines += ["", "### by_source_and_factor_level", "", "| source | factor level | G-Mean² |", "|---|---|---:|"]
            for item in summary["by_source_and_factor_level"]:
                final = item.get("final_summary", [])
                values = [row.get("gmean2", row.get("gmean2_mean")) for row in final if isinstance(row, dict)] if isinstance(final, list) else []
                values = [float(value) for value in values if value is not None]
                lines.append(f"| {item.get('source_id', 'N/A')} | {item.get('factor_level', 'N/A')} | {_fmt(sum(values) / len(values) if values else None)} |")
        lines += ["", "```json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), "```"]
        write_text(path, "\n".join(lines) + "\n")
        return
    lines.append("| {rows} | {pos} | {neg} | {tpr} | {tnr} | {g} | {invalid} |".format(
        rows=overall.get("rows", "N/A"), pos=overall.get("positive_rows", "N/A"),
        neg=overall.get("negative_rows", "N/A"), tpr=_fmt(overall.get("tpr")),
        tnr=_fmt(overall.get("tnr")), g=_fmt(overall.get("gmean2")), invalid=overall.get("invalid", "N/A")))
    for name in ("by_factor_level", "by_source", "by_family"):
        groups = summary.get("slices", {}).get(name, {})
        if not groups: continue
        lines += ["", f"## {name}", "", "| group | rows | TPR | TNR | G-Mean² | invalid |", "|---|---:|---:|---:|---:|---:|"]
        for key, row in groups.items():
            lines.append(f"| {key} | {row.get('rows', 'N/A')} | {_fmt(row.get('tpr'))} | {_fmt(row.get('tnr'))} | {_fmt(row.get('gmean2'))} | {row.get('invalid', 'N/A')} |")
    if not overall:
        for name in ("factor_level_human_semantic", "trajectory_any_path_human_semantic"):
            groups = summary.get(name, {})
            if groups:
                lines += ["", f"## {name}", "", "| factor level | rows | TPR | TNR | G-Mean² | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|---:|"]
                for key, row in groups.items():
                    lines.append(f"| {key} | {row.get('rows', row.get('trajectories_by_level', 'N/A'))} | {_fmt(row.get('tpr'))} | {_fmt(row.get('tnr'))} | {_fmt(row.get('gmean2'))} | {_fmt(row.get('auroc'))} | {_fmt(row.get('auprc'))} |")
                break
    write_text(path, "\n".join(lines) + "\n")


def write_suite_report(summary: dict[str, Any], path) -> None:
    level_macro = summary.get("macro_gmean2_by_factor_level", {})
    lines = ["# Factor Monitor Evaluation Suite", "", f"Manifest: `{summary.get('manifest', 'N/A')}`", "", f"Formal metric: `G-Mean²`; macro: `{_fmt(summary.get('macro_gmean2'))}`", f"L1 macro: `{_fmt(level_macro.get('L1'))}`; L3 macro: `{_fmt(level_macro.get('L3'))}`", "", "| benchmark | track | status | G-Mean² | AUROC | AUPRC | error |", "|---|---|---|---:|---:|---:|---|"]
    for result in summary.get("results", []):
        overall = _overall(result.get("summary", {}))
        if not overall:
            summary = result.get("summary", {})
            values = []
            for item in summary.get("by_source_and_factor_level", []):
                final = item.get("final_summary", [])
                if isinstance(final, list):
                    values.extend(row.get("gmean2", row.get("gmean2_mean")) for row in final if isinstance(row, dict))
            for key in ("factor_level_human_semantic", "trajectory_any_path_human_semantic"):
                values.extend(row.get("gmean2") for row in summary.get(key, {}).values() if isinstance(row, dict))
            values = [float(value) for value in values if value is not None]
            if values:
                overall = {"gmean2": sum(values) / len(values)}
        lines.append(f"| {result.get('id')} | {result.get('track')} | {result.get('status')} | {_fmt(overall.get('gmean2'))} | {_fmt(overall.get('auroc'))} | {_fmt(overall.get('auprc'))} | {result.get('error', '')} |")
    write_text(path, "\n".join(lines) + "\n")
