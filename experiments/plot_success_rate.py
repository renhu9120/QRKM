from __future__ import annotations

"""
Fig. 2 of the letter: success rate and mean epochs against the oversampling ratio n/d.

For each run, saves **three** PDFs: ``success_rate``, ``mean_steps``, and ``mean_time_s`` vs
``n/d``. When ``SHOW_PREVIEW`` is True, opens one window with ``subplot(1, 3)`` for all three.

Edit ``SUCCESS_RATE_LIST`` and ``# === RUN CONFIG`` below. Paths are resolved relative to
``PROJECT_ROOT`` when not absolute. Every row of a CSV is a measured node; the sweep is run
in decreasing ``n/d`` and stops at the first ratio with zero successes. Rows with
``trials == 0`` are rejected: such rows would be never-evaluated placeholders, not data.

With ``EXTEND_ZERO_TAIL`` the drawn curve is continued from that terminal zero down to the
low end of the protocol grid, so that every curve spans the full tested interval and the
reader can see where each method fails rather than where its sweep happened to stop. This
is a *plotting* decision only: nothing is written back to the result CSVs, the extension is
applied solely below a node that was measured to have zero successes, and the mean-time
curve is left undefined there (no timing is invented). The figure caption states it.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.hyperparams as hp


def default_param_list() -> "np.ndarray":
    """Oversampling grid of the letter, on an integer tick scale.

    Nodes: 2, 3, 3.6, 3.8, ..., 6.8, 7, 8, ..., 12 (25 nodes). The ticks are integers
    divided by 10 rather than ``np.arange`` outputs, so ``n`` is free of float drift.

    The sweep is evaluated in *decreasing* ``n/d`` and terminates at the first ratio
    with zero successes: recovery is empirically monotone in ``n/d``, high ratios are
    cheap because every trial converges quickly, and the ratios below the first zero
    would each burn the full epoch budget while carrying no information. Nodes below
    the stopping point are simply not written -- they were not measured.
    """
    coarse_lo = np.array([20, 30], dtype=float) / 10.0            # 2, 3
    fine = np.arange(36, 69, 2, dtype=float) / 10.0               # 3.6, 3.8, ..., 6.8
    coarse_hi = np.arange(70, 121, 10, dtype=float) / 10.0        # 7, 8, ..., 12
    return np.concatenate([coarse_lo, fine, coarse_hi])

SeriesRow = Dict[str, Any]
PreparedSeries = Dict[str, Any]

# Draw each curve across the whole protocol grid by continuing it below its terminal
# zero-success node (see ``_extend_zero_tail``). Display only; the CSVs are untouched.
EXTEND_ZERO_TAIL: bool = True

# Pure line styles accepted by ``linestyle=``.
_PURE_LINE_STYLES = {
    "-",
    "--",
    "-.",
    ":",
    "None",
    " ",
    "",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
}

_METRIC_KEYS: Tuple[str, ...] = ("success_rate", "mean_steps", "mean_time_s")


def _resolve_csv_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_csv_d(csv_path: Path) -> Optional[int]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "d" not in reader.fieldnames:
            return None
        ds: set[int] = set()
        for row in reader:
            try:
                ds.add(int(float(row["d"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(ds) == 1:
        return int(next(iter(ds)))
    if len(ds) > 1:
        return None
    return None


def _validate_series_d_consistency(series_list: List[SeriesRow]) -> None:
    """Warn when overlay CSVs use different signal dimensions ``d``."""
    d_by_label: Dict[str, Optional[int]] = {}
    for entry in series_list:
        label = str(entry.get("label", "method"))
        explicit = entry.get("d", None)
        if explicit is not None:
            d_by_label[label] = int(explicit)
            continue
        csv_path = _resolve_csv_path(str(entry["path"]))
        d_by_label[label] = _read_csv_d(csv_path)

    known = {k: v for k, v in d_by_label.items() if v is not None}
    if len({v for v in known.values()}) > 1:
        detail = ", ".join(f"{k}: d={v}" for k, v in sorted(known.items()))
        print(
            "[plot_success_rate] WARNING: overlay series use different d values; "
            f"curves are not directly comparable ({detail})."
        )


def _load_sweep_columns(csv_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns ``nd`` ascending and a dict of aligned float arrays for each metric column.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    required = {"nd_rate", "success_rate", "mean_steps", "mean_time_s", "trials"}
    nd_list: list[float] = []
    cols: Dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path}: expected columns including {sorted(required)}, got {reader.fieldnames!r}"
            )
        for row in reader:
            try:
                n_trials = int(float(row["trials"]))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{csv_path}: bad trials in row {row!r}") from e
            if n_trials <= 0:
                raise ValueError(
                    f"{csv_path}: row with trials=0 at n/d={row.get('nd_rate')!r}. "
                    "Such rows are never-evaluated placeholders and must not be plotted."
                )
            try:
                nd_list.append(float(row["nd_rate"]))
                for k in _METRIC_KEYS:
                    cols[k].append(float(row[k]))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{csv_path}: bad numeric field in row {row!r}") from e

    if not nd_list:
        raise ValueError(f"{csv_path}: no parsable data rows")

    order = np.argsort(np.asarray(nd_list, dtype=float))
    nd = np.asarray(nd_list, dtype=float)[order]
    out_arrays = {k: np.asarray(cols[k], dtype=float)[order] for k in _METRIC_KEYS}
    return nd, out_arrays


def _extend_zero_tail(
        nd: np.ndarray,
        arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Continue a curve below its terminal zero-success node, for display only.

    The sweep descends in ``n/d`` and exits at the first ratio with zero successes; lower
    ratios are never evaluated, which would leave each curve ending at a different place.
    Here the curve is continued onto the remaining nodes of the protocol grid at success
    rate ``0`` and the full epoch budget, so that the figure shows the whole tested
    interval.  Guarded three ways: it applies only when the lowest *measured* node really
    did have zero successes, it never touches the CSVs, and the wall-clock series is set
    to NaN (an unmeasured timing is left undrawn rather than invented).
    """
    if nd.size == 0 or float(arrays["success_rate"][0]) > 0.0:
        return nd, arrays
    grid = default_param_list()
    add = np.asarray([g for g in grid if g < float(nd[0]) - 1e-9], dtype=float)
    if add.size == 0:
        return nd, arrays
    filler = {
        "success_rate": np.zeros_like(add),
        "mean_steps": np.full_like(add, float(hp.EPOCH_BUDGET)),
        "mean_time_s": np.full_like(add, np.nan),
    }
    out = {k: np.concatenate([filler[k], arrays[k]]) for k in _METRIC_KEYS}
    return np.concatenate([add, nd]), out


def _prepare_series_list(series_list: List[SeriesRow]) -> List[PreparedSeries]:
    prepared: List[PreparedSeries] = []
    for entry in series_list:
        csv_path = _resolve_csv_path(str(entry["path"]))
        nd, arrays = _load_sweep_columns(csv_path)
        if EXTEND_ZERO_TAIL:
            nd, arrays = _extend_zero_tail(nd, arrays)
        row = dict(entry)
        row["nd"] = nd
        row["_y"] = arrays
        prepared.append(row)
    return prepared


def _plot_one_series_on_ax(
        ax: plt.Axes,
        nd: np.ndarray,
        y: np.ndarray,
        *,
        label: str,
        linestyle: str,
        color: Optional[str],
        linewidth: float,
        markersize: float,
        markevery: Optional[int],
        marker: Optional[str],
) -> None:
    ls = str(linestyle).strip()
    kw: dict[str, Any] = {"label": label, "linewidth": linewidth, "markersize": markersize}
    if color is not None:
        kw["color"] = color
    if markevery is not None:
        kw["markevery"] = markevery
    if marker is not None:
        kw["marker"] = marker

    # If ls is a pure linestyle (or marker is explicitly provided), use keyword path.
    # Otherwise treat ls as matplotlib fmt, e.g. "-*", "-x", "--o".
    if (ls in _PURE_LINE_STYLES) or (marker is not None):
        kw["linestyle"] = ls
        ax.plot(nd, y, **kw)
    else:
        ax.plot(nd, y, ls, **kw)


def _apply_metric_axes(
        ax: plt.Axes,
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        ylim: Optional[Tuple[float, float]],
        xlim: Optional[Tuple[float, float]],
        grid: bool,
        legend_loc: str,
        legend_frameon: bool,
        legend_fontsize: float,
        legend_handlelength: float,
        legend_labelspacing: float,
        legend_borderpad: float,
        show_legend: bool,
) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if xlim is not None:
        # Fixed horizontal span, independent of where each sweep terminated. Curves end
        # at their last *measured* node; the axis range is a presentation choice.
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    if show_legend:
        ax.legend(
            loc=str(legend_loc),
            fontsize=float(legend_fontsize),
            frameon=bool(legend_frameon),
            handlelength=float(legend_handlelength),
            labelspacing=float(legend_labelspacing),
            borderpad=float(legend_borderpad),
        )


def _plot_metric_on_ax(
        ax: plt.Axes,
        prepared: List[PreparedSeries],
        y_key: str,
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        ylim: Optional[Tuple[float, float]],
        xlim: Optional[Tuple[float, float]] = None,
        grid: bool = True,
        legend_loc: str,
        legend_frameon: bool,
        legend_fontsize: float,
        legend_handlelength: float,
        legend_labelspacing: float,
        legend_borderpad: float,
        show_legend: bool,
        linewidth: float,
        markersize: float,
        markevery: Optional[int],
) -> None:
    for entry in prepared:
        label = str(entry.get("label", "method"))
        nd = entry["nd"]
        y = entry["_y"][y_key]
        linestyle = str(entry.get("linestyle", entry.get("style", "-")))
        color = entry.get("color", None)
        lw = float(entry.get("linewidth", linewidth))
        ms = float(entry.get("markersize", markersize))
        me = entry.get("markevery", markevery)
        me_i = int(me) if me is not None else None
        mk = entry.get("marker", None)
        _plot_one_series_on_ax(
            ax,
            nd,
            y,
            label=label,
            linestyle=linestyle,
            color=color if color is not None else None,
            linewidth=lw,
            markersize=ms,
            markevery=me_i,
            marker=mk if mk is not None else None,
        )
    _apply_metric_axes(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        ylim=ylim,
        xlim=xlim,
        grid=grid,
        legend_loc=legend_loc,
        legend_frameon=legend_frameon,
        legend_fontsize=legend_fontsize,
        legend_handlelength=legend_handlelength,
        legend_labelspacing=legend_labelspacing,
        legend_borderpad=legend_borderpad,
        show_legend=show_legend,
    )


def plot_sr_metrics(
        series_list: List[SeriesRow],
        *,
        stem: Path,
        xlabel: str,
        titles: Dict[str, str],
        ylabels: Dict[str, str],
        ylims: Dict[str, Optional[Tuple[float, float]]],
        xlim: Optional[Tuple[float, float]] = None,
        figsize_single: Tuple[float, float],
        figsize_preview: Tuple[float, float],
        dpi: int,
        grid: bool,
        legend_loc: str,
        legend_frameon: bool,
        legend_fontsize: float,
        legend_handlelength: float,
        legend_labelspacing: float,
        legend_borderpad: float,
        show_preview: bool,
        linewidth: float,
        markersize: float,
        markevery: Optional[int],
        preview_legend_subplot: int,
) -> None:
    prepared = _prepare_series_list(series_list)
    stem.parent.mkdir(parents=True, exist_ok=True)

    for y_key in _METRIC_KEYS:
        fig, ax = plt.subplots(figsize=figsize_single, dpi=dpi)
        _plot_metric_on_ax(
            ax,
            prepared,
            y_key,
            xlabel=xlabel,
            ylabel=str(ylabels[y_key]),
            title=str(titles.get(y_key, "")),
            ylim=ylims.get(y_key),
            xlim=xlim,
            grid=grid,
            legend_loc=legend_loc,
            legend_frameon=legend_frameon,
            legend_fontsize=legend_fontsize,
            legend_handlelength=legend_handlelength,
            legend_labelspacing=legend_labelspacing,
            legend_borderpad=legend_borderpad,
            show_legend=True,
            linewidth=linewidth,
            markersize=markersize,
            markevery=markevery,
        )
        out_pdf = stem.parent / f"{stem.name}_{y_key}.pdf"
        fig.tight_layout()
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"[plot_success_rate] saved: {out_pdf}")
        plt.close(fig)

    if show_preview:
        fig_p, axes = plt.subplots(1, 3, figsize=figsize_preview, dpi=dpi)
        for idx, y_key in enumerate(_METRIC_KEYS):
            ax = axes[idx]
            _plot_metric_on_ax(
                ax,
                prepared,
                y_key,
                xlabel=xlabel,
                ylabel=str(ylabels[y_key]),
                title=str(titles.get(y_key, "")),
                ylim=ylims.get(y_key),
                xlim=xlim,
                grid=grid,
                legend_loc=legend_loc,
                legend_frameon=legend_frameon,
                legend_fontsize=legend_fontsize,
                legend_handlelength=legend_handlelength,
                legend_labelspacing=legend_labelspacing,
                legend_borderpad=legend_borderpad,
                show_legend=(idx == int(preview_legend_subplot)),
                linewidth=linewidth,
                markersize=markersize,
                markevery=markevery,
            )
        fig_p.suptitle("Preview: success rate | mean steps | mean time", fontsize=12, y=1.02)
        fig_p.tight_layout()
        plt.show()
        plt.close(fig_p)


# =============================================================================
# === RUN CONFIG — edit here ==================================================
# =============================================================================
XLABEL = r"$n/d$"
# Per-metric figure title (optional; empty string hides).
TITLES: Dict[str, str] = {
    "success_rate": "",
    "mean_steps": "",
    "mean_time_s": "",
}
YLABELS: Dict[str, str] = {
    "success_rate": "Success rate",
    "mean_steps": "Mean epochs",
    "mean_time_s": "Mean time (s)",
}
# ``None`` → matplotlib autoscale for that axis.
YLIM_SUCCESS_RATE: Optional[Tuple[float, float]] = (-0.02, 1.02)
YLIM_MEAN_STEPS: Optional[Tuple[float, float]] = None
YLIM_MEAN_TIME_S: Optional[Tuple[float, float]] = None
# Horizontal span of the tested protocol; kept fixed so that terminating the
# descending sweep at the first zero-success ratio does not shrink the figure.
XLIM: Optional[Tuple[float, float]] = (2.0, 12.0)

FIGSIZE_SINGLE = (3.6, 3.6)
FIGSIZE_PREVIEW = (11.0, 3.2)
DPI = 300
GRID = True
LEGEND_LOC = "upper left"  # "lower right"
LEGEND_FRAMEON = True
# Smaller legend for dense paper figures (pt).
LEGEND_FONTSIZE = 4.0
# Tighter legend box (matplotlib units; smaller => more compact).
LEGEND_HANDLELENGTH = 1.35
LEGEND_LABELSPACING = 0.22
LEGEND_BORDERPAD = 0.35
SHOW_PREVIEW = False
LINEWIDTH = 1.35
MARKERSIZE = 4.0
MARKEVERY: Optional[int] = 3
# Which subplot index (0, 1, or 2) shows the legend in preview mode (others avoid overlap).
PREVIEW_LEGEND_SUBPLOT = 0

OUT_DIR = PROJECT_ROOT / "output" / "figures"

# Shared-init sweep at d = 100 -- the measured data behind Fig. 2 of the letter.
_FAIR_DIR = "results/success_rate_d100"
SUCCESS_RATE_LIST: List[SeriesRow] = [
    {"label": "QRKM", "path": f"{_FAIR_DIR}/sr_alg_qrkm_cuda_batched_d100.csv", "linestyle": "-", "color": "tab:red"},
    {"label": "QADMM", "path": f"{_FAIR_DIR}/sr_alg_qadmm_cuda_batched_d100.csv", "linestyle": "--", "color": "teal"},
    {"label": "QRAF", "path": f"{_FAIR_DIR}/sr_alg_qraf_cuda_batched_d100.csv", "linestyle": "-x", "color": "tab:blue"},
    {"label": "QARAF", "path": f"{_FAIR_DIR}/sr_alg_qaraf_cuda_batched_d100.csv", "linestyle": "-.", "color": "tab:orange"},
    {"label": "QWF", "path": f"{_FAIR_DIR}/sr_alg_qwf_cuda_batched_d100.csv", "linestyle": "-o", "color": "black"},
    {"label": "QRWF", "path": f"{_FAIR_DIR}/sr_alg_qrwf_cuda_batched_d100.csv", "linestyle": ":", "color": "tab:brown"},
    {"label": "QPAF", "path": f"{_FAIR_DIR}/sr_alg_qpaf_cuda_batched_d100.csv", "linestyle": "-d", "color": "green"},
]


def main() -> None:
    _validate_series_d_consistency(SUCCESS_RATE_LIST)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = OUT_DIR / f"sr_overlay_{stamp}"
    ylims: Dict[str, Optional[Tuple[float, float]]] = {
        "success_rate": YLIM_SUCCESS_RATE,
        "mean_steps": YLIM_MEAN_STEPS,
        "mean_time_s": YLIM_MEAN_TIME_S,
    }
    plot_sr_metrics(
        SUCCESS_RATE_LIST,
        stem=stem,
        xlabel=str(XLABEL),
        titles={k: str(TITLES.get(k, "")) for k in _METRIC_KEYS},
        ylabels={k: str(YLABELS[k]) for k in _METRIC_KEYS},
        ylims=ylims,
        xlim=XLIM,
        figsize_single=(float(FIGSIZE_SINGLE[0]), float(FIGSIZE_SINGLE[1])),
        figsize_preview=(float(FIGSIZE_PREVIEW[0]), float(FIGSIZE_PREVIEW[1])),
        dpi=int(DPI),
        grid=bool(GRID),
        legend_loc=str(LEGEND_LOC),
        legend_frameon=bool(LEGEND_FRAMEON),
        legend_fontsize=float(LEGEND_FONTSIZE),
        legend_handlelength=float(LEGEND_HANDLELENGTH),
        legend_labelspacing=float(LEGEND_LABELSPACING),
        legend_borderpad=float(LEGEND_BORDERPAD),
        show_preview=bool(SHOW_PREVIEW),
        linewidth=float(LINEWIDTH),
        markersize=float(MARKERSIZE),
        markevery=MARKEVERY,
        preview_legend_subplot=int(PREVIEW_LEGEND_SUBPLOT),
    )


if __name__ == "__main__":
    main()
