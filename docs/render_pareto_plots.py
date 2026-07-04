#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_DIR = DOCS / "pareto_plots"

WIDTH = 960
HEIGHT = 620
LEFT = 96
RIGHT = 236
TOP = 76
BOTTOM = 118

PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#4b5563",
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: str
    series: str
    detail: str
    annotate: bool = False


@dataclass(frozen=True)
class CatPoint:
    category: str
    y: float
    label: str
    series: str
    detail: str
    annotate: bool = False


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _read_csv(path: Path, notes: list[str], context: str) -> list[dict[str, str]]:
    if not path.exists():
        notes.append(f"{context}: missing source CSV `{_rel(path)}`.")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        notes.append(f"{context}: source CSV `{_rel(path)}` has no rows.")
    return rows


def _has_columns(
    path: Path,
    rows: list[dict[str, str]],
    columns: list[str],
    notes: list[str],
    context: str,
) -> bool:
    present = set(rows[0]) if rows else set()
    missing = [column for column in columns if column not in present]
    if missing:
        missing_text = ", ".join(f"`{column}`" for column in missing)
        notes.append(f"{context}: `{_rel(path)}` lacks required column(s): {missing_text}.")
        return False
    return True


def _as_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _fmt(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    if abs_value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.5f}"


def _bounds(values: list[float], *, floor_zero: bool = False) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lower = min(values)
    upper = max(values)
    pad = max(abs(lower) * 0.05, 0.05) if lower == upper else (upper - lower) * 0.08
    lower -= pad
    upper += pad
    if floor_zero and lower > 0:
        lower = 0.0
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def _ticks(lower: float, upper: float, count: int = 6) -> list[float]:
    if count <= 1:
        return [lower]
    step = (upper - lower) / (count - 1)
    return [lower + step * index for index in range(count)]


def _series_colors(series_names: list[str]) -> dict[str, str]:
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(series_names)}


def _svg_header(title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{html.escape(title)}</title>",
        f"<desc id=\"desc\">{html.escape(desc)}</desc>",
        "<style>",
        "  .bg { fill: #ffffff; }",
        "  .grid { stroke: #e5e7eb; stroke-width: 1; }",
        "  .axis { stroke: #111827; stroke-width: 1.4; }",
        "  .tick { fill: #374151; font: 12px sans-serif; }",
        "  .title { fill: #111827; font: 700 24px sans-serif; }",
        "  .subtitle { fill: #4b5563; font: 13px sans-serif; }",
        "  .label { fill: #111827; font: 14px sans-serif; }",
        "  .point-label { fill: #111827; font: 10px sans-serif; }",
        "  .legend { fill: #111827; font: 12px sans-serif; }",
        "  .point { stroke: #ffffff; stroke-width: 1.5; }",
        "</style>",
        '<rect class="bg" x="0" y="0" width="960" height="620"/>',
    ]


def _render_legend(parts: list[str], series_names: list[str], colors: dict[str, str]) -> None:
    legend_x = WIDTH - RIGHT + 56
    legend_y = TOP + 8
    parts.append(f'<text class="label" x="{legend_x}" y="{legend_y - 22}">Series</text>')
    for index, series in enumerate(series_names):
        y = legend_y + index * 22
        color = colors[series]
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(
            f'<text class="legend" x="{legend_x + 12}" y="{y + 4}">'
            f"{html.escape(series)}</text>"
        )


def render_scatter(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    points: list[Point],
    desc: str,
    floor_y_zero: bool = False,
) -> str:
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    x_lower, x_upper = _bounds([point.x for point in points])
    y_lower, y_upper = _bounds([point.y for point in points], floor_zero=floor_y_zero)

    def sx(value: float) -> float:
        return LEFT + (value - x_lower) / (x_upper - x_lower) * plot_width

    def sy(value: float) -> float:
        return TOP + (y_upper - value) / (y_upper - y_lower) * plot_height

    series_names = sorted({point.series for point in points})
    colors = _series_colors(series_names)

    parts = _svg_header(title, desc)
    parts.append(f'<text class="title" x="{LEFT}" y="38">{html.escape(title)}</text>')
    parts.append(f'<text class="subtitle" x="{LEFT}" y="60">{html.escape(subtitle)}</text>')

    for value in _ticks(x_lower, x_upper):
        x = sx(value)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{TOP + plot_height}"/>')
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{TOP + plot_height + 22}" '
            f'text-anchor="middle">{html.escape(_fmt(value))}</text>'
        )
    for value in _ticks(y_lower, y_upper):
        y = sy(value)
        parts.append(f'<line class="grid" x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_width}" y2="{y:.2f}"/>')
        parts.append(
            f'<text class="tick" x="{LEFT - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f"{html.escape(_fmt(value))}</text>"
        )

    parts.append(f'<line class="axis" x1="{LEFT}" y1="{TOP + plot_height}" x2="{LEFT + plot_width}" y2="{TOP + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_height}"/>')
    parts.append(
        f'<text class="label" x="{LEFT + plot_width / 2:.2f}" y="{HEIGHT - 38}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text class="label" transform="translate(28 {TOP + plot_height / 2:.2f}) '
        f'rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )

    for point in points:
        x = sx(point.x)
        y = sy(point.y)
        color = colors[point.series]
        title_text = (
            f"{point.label}: {x_label}={_fmt(point.x)}, {y_label}={_fmt(point.y)}. "
            f"{point.detail}"
        )
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}">')
        parts.append(f"<title>{html.escape(title_text)}</title>")
        parts.append("</circle>")
        if point.annotate:
            parts.append(
                f'<text class="point-label" x="{x + 8:.2f}" y="{y - 8:.2f}">'
                f"{html.escape(point.label)}</text>"
            )

    _render_legend(parts, series_names, colors)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_categorical(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    points: list[CatPoint],
    desc: str,
    floor_y_zero: bool = True,
) -> str:
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    categories = list(dict.fromkeys(point.category for point in points))
    y_lower, y_upper = _bounds([point.y for point in points], floor_zero=floor_y_zero)

    def sx(category: str, offset: float = 0.0) -> float:
        if len(categories) == 1:
            return LEFT + plot_width / 2 + offset
        index = categories.index(category)
        return LEFT + index / (len(categories) - 1) * plot_width + offset

    def sy(value: float) -> float:
        return TOP + (y_upper - value) / (y_upper - y_lower) * plot_height

    series_names = sorted({point.series for point in points})
    colors = _series_colors(series_names)

    grouped: dict[str, list[CatPoint]] = {}
    for point in points:
        grouped.setdefault(point.category, []).append(point)

    parts = _svg_header(title, desc)
    parts.append(f'<text class="title" x="{LEFT}" y="38">{html.escape(title)}</text>')
    parts.append(f'<text class="subtitle" x="{LEFT}" y="60">{html.escape(subtitle)}</text>')

    for category in categories:
        x = sx(category)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{TOP + plot_height}"/>')
        parts.append(
            f'<text class="tick" transform="translate({x:.2f} {TOP + plot_height + 24}) '
            f'rotate(-22)" text-anchor="end">{html.escape(category)}</text>'
        )
    for value in _ticks(y_lower, y_upper):
        y = sy(value)
        parts.append(f'<line class="grid" x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_width}" y2="{y:.2f}"/>')
        parts.append(
            f'<text class="tick" x="{LEFT - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f"{html.escape(_fmt(value))}</text>"
        )

    parts.append(f'<line class="axis" x1="{LEFT}" y1="{TOP + plot_height}" x2="{LEFT + plot_width}" y2="{TOP + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_height}"/>')
    parts.append(
        f'<text class="label" x="{LEFT + plot_width / 2:.2f}" y="{HEIGHT - 38}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text class="label" transform="translate(28 {TOP + plot_height / 2:.2f}) '
        f'rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )

    for category, category_points in grouped.items():
        count = len(category_points)
        for index, point in enumerate(category_points):
            offset = (index - (count - 1) / 2.0) * 16.0
            x = sx(category, offset)
            y = sy(point.y)
            color = colors[point.series]
            title_text = f"{point.label}: {y_label}={_fmt(point.y)}. {point.detail}"
            parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}">')
            parts.append(f"<title>{html.escape(title_text)}</title>")
            parts.append("</circle>")
            if point.annotate:
                parts.append(
                    f'<text class="point-label" x="{x + 8:.2f}" y="{y - 8:.2f}">'
                    f"{html.escape(point.label)}</text>"
                )

    _render_legend(parts, series_names, colors)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _short_case(name: str) -> str:
    replacements = {
        "none_routing_only": "none routing",
        "shared_one_per_node": "shared one",
        "shared_half_fraction": "shared half",
        "split_one_per_node": "split one",
        "split_half_fraction": "split half",
    }
    return replacements.get(name, name.replace("_", " "))


def _row_detail(path: Path, row: dict[str, str], columns: list[str]) -> str:
    parts = [f"source={_rel(path)}"]
    for column in columns:
        if row.get(column, ""):
            parts.append(f"{column}={row[column]}")
    return "; ".join(parts)


def _t20_points(
    rows: list[dict[str, str]],
    path: Path,
    x_column: str,
    y_column: str,
    notes: list[str],
    context: str,
) -> list[Point]:
    required = [x_column, y_column, "case_name", "route_row_role"]
    if not _has_columns(path, rows, required, notes, context):
        return []
    points: list[Point] = []
    for row in rows:
        x = _as_float(row, x_column)
        y = _as_float(row, y_column)
        if x is None or y is None:
            continue
        label = _short_case(row["case_name"])
        points.append(
            Point(
                x=x,
                y=y,
                label=label,
                series=row["route_row_role"],
                detail=_row_detail(
                    path,
                    row,
                    [
                        "case_name",
                        "final_nmse",
                        "tokens_per_second",
                        "active_rows_per_token",
                        "validation_accuracy_after_replacement",
                    ],
                ),
                annotate=True,
            )
        )
    return points


def _numeric_points(
    rows: list[dict[str, str]],
    path: Path,
    x_column: str,
    y_column: str,
    series: str,
    label_column: str,
    notes: list[str],
    context: str,
    *,
    annotate: bool = False,
) -> list[Point]:
    required = [x_column, y_column, label_column]
    if not _has_columns(path, rows, required, notes, context):
        return []
    points: list[Point] = []
    for row in rows:
        x = _as_float(row, x_column)
        y = _as_float(row, y_column)
        if x is None or y is None:
            continue
        points.append(
            Point(
                x=x,
                y=y,
                label=row[label_column],
                series=series,
                detail=_row_detail(path, row, [label_column, x_column, y_column]),
                annotate=annotate,
            )
        )
    return points


def _write_missing_notes(notes: list[str]) -> str:
    lines = [
        "# Pareto Plot Missing Sources",
        "",
        "Generated by `docs/render_pareto_plots.py` from committed CSV summaries.",
        "No synthetic or hand-entered metrics were used.",
        "",
        "## Missing Or Limited Source Data",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No missing plot sources were detected.")
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            ".venv/bin/python docs/render_pareto_plots.py",
            ".venv/bin/python docs/render_pareto_plots.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_readme() -> str:
    return "\n".join(
        [
            "# Pareto Plots",
            "",
            "These rendered plots are generated from committed CSV summaries only.",
            "They are validation and smoke-study artifacts, not final FFF-student claims.",
            "",
            "| Plot | Committed source |",
            "| --- | --- |",
            "| [Accuracy vs active rows](accuracy_vs_active_rows.svg) | `docs/t20_route_row_output_ablation_results.csv` |",
            "| [Accuracy vs throughput](accuracy_vs_throughput.svg) | `docs/t20_route_row_output_ablation_results.csv` |",
            "| [MSE vs active rows](mse_vs_active_rows.svg) | `docs/t20_route_row_output_ablation_results.csv`, `docs/t13_stage_f_layerwise_summary.csv` |",
            "| [Dead leaves vs balance](dead_leaves_vs_balance.svg) | `docs/t13_stage_d_arch_summary.csv` |",
            "| [Route entropy vs accuracy](route_entropy_vs_accuracy.svg) | `docs/t20_route_row_output_ablation_results.csv` |",
            "| [STE method vs MSE/throughput](ste_method_vs_mse_throughput.svg) | `docs/t13_stage_c_router_summary.csv`, `docs/t13_stage_d_arch_summary.csv`, `docs/t13_stage_f_layerwise_summary.csv` |",
            "",
            "Missing or limited plot-source coverage is documented in",
            "[missing_sources.md](missing_sources.md).",
            "",
            "Regenerate and validate:",
            "",
            "```bash",
            ".venv/bin/python docs/render_pareto_plots.py",
            ".venv/bin/python docs/render_pareto_plots.py --check",
            "```",
            "",
        ]
    )


def build_outputs() -> dict[Path, str]:
    notes: list[str] = [
        "Accuracy plots use `validation_accuracy_after_replacement` from the T20 one-layer "
        "route-output sweep. No committed full FFF-student accuracy-vs-active-rows or "
        "accuracy-vs-throughput CSV exists.",
        "`docs/t13_stage_f_layerwise_summary.csv` has route entropy and MSE columns but no "
        "validation accuracy column, so route-entropy-vs-accuracy uses T20 one-layer "
        "replacement accuracy only.",
        "`docs/t13_stage_f_layerwise_summary.csv` has dead-leaf counts but no "
        "`balance_recipe` or `balance_coeff` columns, so dead-leaves-vs-balance uses the "
        "Stage D architecture sweep only. No all-layer balance sweep CSV is committed.",
        "`docs/t13_stage_e_layer_summary.csv` has MSE/throughput columns but no "
        "`router_recipe` or STE-method column, so it is excluded from the STE-method plot.",
    ]

    t20_path = DOCS / "t20_route_row_output_ablation_results.csv"
    stage_c_path = DOCS / "t13_stage_c_router_summary.csv"
    stage_d_path = DOCS / "t13_stage_d_arch_summary.csv"
    stage_f_path = DOCS / "t13_stage_f_layerwise_summary.csv"

    t20_rows = _read_csv(t20_path, notes, "T20 route-output plots")
    stage_c_rows = _read_csv(stage_c_path, notes, "Stage C STE-method plot")
    stage_d_rows = _read_csv(stage_d_path, notes, "Stage D balance and method plots")
    stage_f_rows = _read_csv(stage_f_path, notes, "Stage F layerwise plots")

    outputs: dict[Path, str] = {}

    acc_active = _t20_points(
        t20_rows,
        t20_path,
        "active_rows_per_token",
        "validation_accuracy_after_replacement",
        notes,
        "Accuracy vs active rows",
    )
    outputs[OUT_DIR / "accuracy_vs_active_rows.svg"] = render_scatter(
        title="Accuracy vs Active Rows",
        subtitle="T20 one-layer route-output replacement sweep",
        x_label="Active rows per token",
        y_label="Validation accuracy after replacement",
        points=acc_active,
        desc="T20 route-row output ablation validation accuracy against active rows.",
    )

    acc_throughput = _t20_points(
        t20_rows,
        t20_path,
        "tokens_per_second",
        "validation_accuracy_after_replacement",
        notes,
        "Accuracy vs throughput",
    )
    outputs[OUT_DIR / "accuracy_vs_throughput.svg"] = render_scatter(
        title="Accuracy vs Throughput",
        subtitle="T20 one-layer route-output replacement sweep",
        x_label="Tokens per second",
        y_label="Validation accuracy after replacement",
        points=acc_throughput,
        desc="T20 route-row output ablation validation accuracy against throughput.",
    )

    mse_points = _numeric_points(
        t20_rows,
        t20_path,
        "active_rows_per_token",
        "final_nmse",
        "T20 route-output sweep",
        "case_name",
        notes,
        "MSE vs active rows, T20",
        annotate=True,
    )
    mse_points += _numeric_points(
        stage_f_rows,
        stage_f_path,
        "active_rows_per_token",
        "final_nmse",
        "T13 Stage F layerwise",
        "layer",
        notes,
        "MSE vs active rows, Stage F",
    )
    outputs[OUT_DIR / "mse_vs_active_rows.svg"] = render_scatter(
        title="MSE vs Active Rows",
        subtitle="T20 route-output cases plus all Stage F layerwise distillations",
        x_label="Active rows per token",
        y_label="Final normalized MSE",
        points=mse_points,
        desc="Final normalized MSE against active rows from T20 and Stage F CSVs.",
        floor_y_zero=True,
    )

    dead_balance: list[CatPoint] = []
    if _has_columns(
        stage_d_path,
        stage_d_rows,
        ["balance_recipe", "balance_coeff", "dead_leaves", "router_recipe", "depth"],
        notes,
        "Dead leaves vs balance",
    ):
        for row in stage_d_rows:
            dead = _as_float(row, "dead_leaves")
            if dead is None:
                continue
            category = f"{row['balance_recipe']} coeff={row['balance_coeff']}"
            label = f"{row['router_recipe']} d{row['depth']}"
            dead_balance.append(
                CatPoint(
                    category=category,
                    y=dead,
                    label=label,
                    series=row["router_recipe"],
                    detail=_row_detail(
                        stage_d_path,
                        row,
                        [
                            "balance_recipe",
                            "balance_coeff",
                            "final_nmse",
                            "tokens_per_second",
                            "dead_leaves",
                        ],
                    ),
                )
            )
    outputs[OUT_DIR / "dead_leaves_vs_balance.svg"] = render_categorical(
        title="Dead Leaves vs Balance",
        subtitle="Stage D architecture sweep categories by balance recipe and coefficient",
        x_label="Balance recipe and coefficient",
        y_label="Dead leaves",
        points=dead_balance,
        desc="Dead leaves by balance recipe and balance coefficient from Stage D.",
    )

    entropy_accuracy = _t20_points(
        t20_rows,
        t20_path,
        "route_entropy_mean",
        "validation_accuracy_after_replacement",
        notes,
        "Route entropy vs accuracy",
    )
    outputs[OUT_DIR / "route_entropy_vs_accuracy.svg"] = render_scatter(
        title="Route Entropy vs Accuracy",
        subtitle="T20 one-layer route-output replacement sweep",
        x_label="Mean route entropy",
        y_label="Validation accuracy after replacement",
        points=entropy_accuracy,
        desc="T20 validation accuracy against mean route entropy.",
    )

    ste_points = _numeric_points(
        stage_c_rows,
        stage_c_path,
        "tokens_per_second",
        "final_nmse",
        "Stage C router recipe",
        "router_recipe",
        notes,
        "STE method vs MSE/throughput, Stage C",
        annotate=True,
    )
    if _has_columns(
        stage_d_path,
        stage_d_rows,
        ["tokens_per_second", "final_nmse", "router_recipe", "depth", "route_row_role"],
        notes,
        "STE method vs MSE/throughput, Stage D",
    ):
        for row in stage_d_rows:
            x = _as_float(row, "tokens_per_second")
            y = _as_float(row, "final_nmse")
            if x is None or y is None:
                continue
            label = f"{row['router_recipe']} d{row['depth']}"
            ste_points.append(
                Point(
                    x=x,
                    y=y,
                    label=label,
                    series="Stage D architecture sweep",
                    detail=_row_detail(
                        stage_d_path,
                        row,
                        ["router_recipe", "depth", "route_row_role", "final_nmse"],
                    ),
                )
            )
    if _has_columns(
        stage_f_path,
        stage_f_rows,
        ["tokens_per_second", "final_nmse", "router_recipe"],
        notes,
        "STE method vs MSE/throughput, Stage F",
    ):
        stage_f_x = [_as_float(row, "tokens_per_second") for row in stage_f_rows]
        stage_f_y = [_as_float(row, "final_nmse") for row in stage_f_rows]
        stage_f_x_values = [value for value in stage_f_x if value is not None]
        stage_f_y_values = [value for value in stage_f_y if value is not None]
        if stage_f_x_values and stage_f_y_values:
            ste_points.append(
                Point(
                    x=mean(stage_f_x_values),
                    y=mean(stage_f_y_values),
                    label="Stage F vanilla mean",
                    series="Stage F layerwise mean",
                    detail=(
                        f"source={_rel(stage_f_path)}; rows={len(stage_f_rows)}; "
                        f"router_recipe=vanilla_ste"
                    ),
                    annotate=True,
                )
            )
    outputs[OUT_DIR / "ste_method_vs_mse_throughput.svg"] = render_scatter(
        title="STE Method vs MSE/Throughput",
        subtitle="Stage C recipes, Stage D architecture sweep, and Stage F vanilla mean",
        x_label="Tokens per second",
        y_label="Final normalized MSE",
        points=ste_points,
        desc="Router or STE method final normalized MSE against throughput.",
        floor_y_zero=True,
    )

    outputs[OUT_DIR / "missing_sources.md"] = _write_missing_notes(notes)
    outputs[OUT_DIR / "README.md"] = _write_readme()
    return outputs


def write_outputs(*, check: bool) -> int:
    outputs = build_outputs()
    if check:
        stale: list[str] = []
        for path, expected in outputs.items():
            if not path.exists():
                stale.append(f"missing: {_rel(path)}")
                continue
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                stale.append(f"stale: {_rel(path)}")
        if stale:
            for item in stale:
                print(item)
            return 1
        print(f"pareto plot artifacts are current: {_rel(OUT_DIR)}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(outputs)} pareto plot artifacts to {_rel(OUT_DIR)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed plots are stale")
    args = parser.parse_args()
    return write_outputs(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
