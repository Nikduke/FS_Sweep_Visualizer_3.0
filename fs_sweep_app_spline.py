import os
import hashlib
import json
from typing import Dict, List, Tuple, Optional

# Main app baseline with client-side zoom persistence and JS-side interactive case controls.

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType


# ---- Page config ----
st.set_page_config(page_title="FS Sweep Visualizer (Spline)", layout="wide")

# =============================================================================
# Settings (single place to tune defaults)
#
# Preference: use named constants grouped by purpose (lowest code churn and most
# readable in a single-file Streamlit app). Keep `STYLE` as a dict because it
# maps directly to Plotly layout fields.
# =============================================================================

# ---- Style (applies to on-page AND exports) ----
STYLE = {
    "font_family": "Open Sans, verdana, arial, sans-serif",
    "font_color": "#000000",
    "base_font_size_px": 14,
    "tick_font_size_px": 14,
    "axis_title_font_size_px": 16,
    "legend_font_size_px": 14,
    "bold_axis_titles": True,
    # Space between tick labels and axis title (px). Set to None to use auto heuristic.
    "xaxis_title_standoff_px": None,
    "yaxis_title_standoff_px": None,
}

# ---- Layout (web view) ----
# NOTE: Keep the bottom legend layout; axis overlap is handled by title standoff + margins.
DEFAULT_FIGURE_WIDTH_PX = 1400
TOP_MARGIN_PX = 40
BOTTOM_AXIS_PX = 60
LEFT_MARGIN_PX = 60
RIGHT_MARGIN_PX = 20

# Layout heuristics (auto margins based on font sizes)
BOTTOM_AXIS_TICK_MULT = 2.4
BOTTOM_AXIS_TITLE_MULT = 1.6
LEFT_MARGIN_TICK_MULT = 4.4
LEFT_MARGIN_TITLE_MULT = 1.6
AXIS_TITLE_STANDOFF_TICK_MULT = 1.1
AXIS_TITLE_STANDOFF_MIN_PX = 10

# ---- Legend sizing (web + export) ----
LEGEND_ROW_HEIGHT_FACTOR = 2.0  # web estimate: row height ~= legend_font_size * factor
LEGEND_PADDING_PX = 18  # extra padding used in legend height estimate and export margin
WEB_LEGEND_EXTRA_PAD_PX = 20  # web-only safety pad to reduce last-row clipping
WEB_LEGEND_MAX_HEIGHT_PX = 1000  # cap reserved web legend height
AUTO_WIDTH_ESTIMATE_PX = 950  # used only when Plotly auto-sizes (legend row estimate)

# ---- Performance / computation ----
DEFAULT_SPLINE_SMOOTHING = 1.0
SPLINE_SMOOTHING_MIN = 0.0
SPLINE_SMOOTHING_MAX = 1.3
SPLINE_SMOOTHING_STEP = 0.05
XR_EPS = 1e-9  # treat |R| < XR_EPS as invalid for X/R
XR_EPS_DISPLAY = "1e-9"  # shown in UI text (keep in sync with XR_EPS)

# ---- Export ----
EXPORT_IMAGE_SCALE = 4  # modebar + full-legend export
EXPORT_DOM_ID_HASH_LEN = 12
EXPORT_FALLBACK_COLOR = "#444"

# Full-legend export (JS layout heuristics)
EXPORT_LEGEND_ROW_HEIGHT_FACTOR = 1.25
EXPORT_SAMPLE_LINE_MIN_PX = 18
EXPORT_SAMPLE_LINE_MULT = 1.8
EXPORT_SAMPLE_GAP_MIN_PX = 6
EXPORT_SAMPLE_GAP_MULT = 0.6
EXPORT_TEXT_PAD_MIN_PX = 8
EXPORT_TEXT_PAD_MULT = 0.8
EXPORT_LEGEND_TAIL_FONT_MULT = 0.35
EXPORT_LEGEND_ROW_Y_OFFSET = 0.6
EXPORT_COL_PADDING_MAX_PX = 12
EXPORT_COL_PADDING_FRAC = 0.06

# ---- App behavior ----
UPLOAD_SHA1_PREFIX_LEN = 10

# Zoom listener (JS bind loop and mount-time relayout ignore window)
ZOOM_BIND_TRIES = 80
ZOOM_BIND_INTERVAL_MS = 100
ZOOM_IGNORE_AUTORANGE_MS = 1200

# ---- Color shading (clustered color palette) ----
COLOR_FALLBACK_RGB255 = (68, 68, 68)
COLOR_LIGHTEN_MAX_T = 0.40
COLOR_DARKEN_MAX_T = 0.25

# ---- Interactive selection styling ----
SELECTED_LINE_WIDTH = 2.5
DIM_LINE_WIDTH = 1.0
DIM_LINE_OPACITY = 0.35
DIM_LINE_COLOR = "#B8B8B8"
SELECTED_MARKER_SIZE = 10.0
DIM_MARKER_OPACITY = 0.28

# ---- R vs X scatter ----
RX_SCATTER_HEIGHT_FACTOR = 1.5

_plotly_relayout_listener = components.declare_component(
    "plotly_relayout_listener",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_relayout_listener")),
)
_plotly_selection_bridge = components.declare_component(
    "plotly_selection_bridge_v14",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_selection_bridge")),
)


def plotly_relayout_listener(
    data_id: str,
    plot_count: int = 3,
    plot_ids: Optional[List[str]] = None,
    debounce_ms: int = 120,
    nonce: int = 0,
    reset_token: int = 0,
) -> Optional[Dict[str, object]]:
    # Client-side zoom persistence: binds to Plotly charts and stores axis ranges
    # in browser localStorage. Returns None (no Python roundtrip on zoom).
    return _plotly_relayout_listener(  # type: ignore[misc]
        data_id=str(data_id),
        plot_count=int(plot_count),
        plot_ids=list(plot_ids or []),
        debounce_ms=int(debounce_ms),
        nonce=int(nonce),
        reset_token=int(reset_token),
        bind_tries=int(ZOOM_BIND_TRIES),
        bind_interval_ms=int(ZOOM_BIND_INTERVAL_MS),
        ignore_autorange_ms=int(ZOOM_IGNORE_AUTORANGE_MS),
        key=f"plotly_relayout_listener:{data_id}",
        default=None,
    )


def plotly_selection_bridge(
    data_id: str,
    chart_id: str,
    plot_ids: List[str],
    cases_meta: List[Dict[str, object]],
    part_labels: List[str],
    color_by_options: List[str],
    color_maps: Dict[str, Dict[str, str]],
    auto_color_part_label: str = "",
    color_by_default: str = "Auto",
    show_only_default: bool = False,
    selected_marker_size: float = float(SELECTED_MARKER_SIZE),
    dim_marker_opacity: float = float(DIM_MARKER_OPACITY),
    selected_line_width: float = float(SELECTED_LINE_WIDTH),
    dim_line_width: float = float(DIM_LINE_WIDTH),
    dim_line_opacity: float = float(DIM_LINE_OPACITY),
    dim_line_color: str = str(DIM_LINE_COLOR),
    f_base: float = 50.0,
    n_min: float = 0.0,
    n_max: float = 1.0,
    show_harmonics_default: bool = True,
    bin_width_hz_default: float = 0.0,
    rx_status_dom_id: str = "",
    rx_freq_steps: int = 0,
    reset_token: int = 0,
    selection_reset_token: int = 0,
    render_nonce: int = 0,
    enable_selection: bool = True,
) -> None:
    _plotly_selection_bridge(  # type: ignore[misc]
        data_id=str(data_id),
        chart_id=str(chart_id),
        plot_ids=list(plot_ids or []),
        cases_meta=list(cases_meta or []),
        part_labels=list(part_labels or []),
        color_by_options=list(color_by_options or []),
        color_maps=dict(color_maps or {}),
        auto_color_part_label=str(auto_color_part_label or ""),
        color_by_default=str(color_by_default),
        show_only_default=bool(show_only_default),
        selected_marker_size=float(selected_marker_size),
        dim_marker_opacity=float(dim_marker_opacity),
        selected_line_width=float(selected_line_width),
        dim_line_width=float(dim_line_width),
        dim_line_opacity=float(dim_line_opacity),
        dim_line_color=str(dim_line_color),
        f_base=float(f_base),
        n_min=float(n_min),
        n_max=float(n_max),
        show_harmonics_default=bool(show_harmonics_default),
        bin_width_hz_default=float(bin_width_hz_default),
        rx_status_dom_id=str(rx_status_dom_id),
        rx_freq_steps=int(rx_freq_steps),
        reset_token=int(reset_token),
        selection_reset_token=int(selection_reset_token),
        render_nonce=int(render_nonce),
        enable_selection=bool(enable_selection),
        key=f"plotly_selection_bridge:{data_id}:{chart_id}",
        height=260,
        default=0,
    )


def _note_upload_change() -> None:
    # Called by st.file_uploader(on_change=...): used to trigger filter+zoom reset on any upload action.
    st.session_state["upload_nonce"] = int(st.session_state.get("upload_nonce", 0)) + 1
    up = st.session_state.get("xlsx_uploader")
    if up is None:
        st.session_state.pop("uploaded_file_sha1_10", None)
        st.session_state.pop("uploaded_file_name", None)
        return
    try:
        st.session_state["uploaded_file_sha1_10"] = hashlib.sha1(up.getvalue()).hexdigest()[: int(UPLOAD_SHA1_PREFIX_LEN)]
        st.session_state["uploaded_file_name"] = getattr(up, "name", None)
    except Exception:
        st.session_state.pop("uploaded_file_sha1_10", None)
        st.session_state.pop("uploaded_file_name", None)


def _clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(val)))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (int(_clamp_int(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    tt = float(max(0.0, min(1.0, t)))
    return (
        int(round(a[0] + (b[0] - a[0]) * tt)),
        int(round(a[1] + (b[1] - a[1]) * tt)),
        int(round(a[2] + (b[2] - a[2]) * tt)),
    )


def _parse_color_to_rgb255(color: str) -> Tuple[int, int, int]:
    """
    Accept Plotly palette entries in either hex ("#rrggbb") or "rgb(...)" / "rgba(...)" form.
    """
    c = str(color).strip()
    if not c:
        return tuple(int(v) for v in COLOR_FALLBACK_RGB255)
    if c.startswith("#"):
        return tuple(int(v) for v in pc.hex_to_rgb(c))
    if c.lower().startswith("rgb"):
        tup = pc.unlabel_rgb(c)
        if len(tup) >= 3:
            return (int(round(tup[0])), int(round(tup[1])), int(round(tup[2])))
    # handle hex without '#'
    c2 = c.lstrip().lower()
    if len(c2) in (3, 6) and all(ch in "0123456789abcdef" for ch in c2):
        if len(c2) == 3:
            c2 = "".join([ch * 2 for ch in c2])
        return tuple(int(v) for v in pc.hex_to_rgb(f"#{c2}"))
    return tuple(int(v) for v in COLOR_FALLBACK_RGB255)


def _shade_hex(base_hex: str, position: float) -> str:
    """
    Create a shade variant of a base color.

    `position` in [-1..1]:
      - negative => darken toward black
      - positive => lighten toward white
    """
    base_rgb = _parse_color_to_rgb255(base_hex)
    p = float(max(-1.0, min(1.0, position)))
    if p >= 0:
        # Lighten
        return _rgb_to_hex(_mix_rgb(base_rgb, (255, 255, 255), t=p * float(COLOR_LIGHTEN_MAX_T)))
    # Darken
    return _rgb_to_hex(_mix_rgb(base_rgb, (0, 0, 0), t=(-p) * float(COLOR_DARKEN_MAX_T)))


def build_clustered_case_colors(cases: List[str], hue_part_override: Optional[int] = None) -> Dict[str, str]:
    """
    Assign colors so related cases cluster by hue, with lighter/darker shades inside each cluster.

    Location suffix (after `__`) is ignored for grouping.
    """
    if not cases:
        return {}

    bases = [split_case_location(c)[0] for c in cases]
    split_parts = [str(b).split("_") for b in bases]
    max_parts = max((len(p) for p in split_parts), default=0)
    if max_parts <= 0:
        # Fallback to simple palette
        palette = pc.qualitative.Safe or pc.qualitative.Plotly or ["#1f77b4"]
        return {
            c: palette[i % len(palette)]
            for i, c in enumerate(sorted(cases))
        }

    # Normalize parts (pad with "")
    parts_norm = [p + [""] * (max_parts - len(p)) for p in split_parts]

    # Pick "hue part":
    # - If hue_part_override is provided and valid, use it.
    # - Otherwise, use the most varying part (ties => earlier part).
    uniq_counts = [len(set(row[i] for row in parts_norm)) for i in range(max_parts)]
    if hue_part_override is not None and 0 <= int(hue_part_override) < int(max_parts):
        hue_part = int(hue_part_override)
        varying = [i for i, n in enumerate(uniq_counts) if n > 1]
        rest = [i for i in varying if i != hue_part]
        shade_part = sorted(rest, key=lambda i: (-uniq_counts[i], i))[0] if rest else None
    else:
        varying = [i for i, n in enumerate(uniq_counts) if n > 1]
        if not varying:
            hue_part = 0
            shade_part = None
        else:
            hue_part = sorted(varying, key=lambda i: (-uniq_counts[i], i))[0]
            rest = [i for i in varying if i != hue_part]
            shade_part = sorted(rest, key=lambda i: (-uniq_counts[i], i))[0] if rest else None

    # Use a combined palette so we have enough distinct hues if there are many groups.
    palette = []
    for pal in (
        getattr(pc.qualitative, "Safe", None),
        getattr(pc.qualitative, "D3", None),
        getattr(pc.qualitative, "Plotly", None),
        getattr(pc.qualitative, "Dark24", None),
        getattr(pc.qualitative, "Light24", None),
    ):
        if pal:
            palette.extend(list(pal))
    if not palette:
        palette = ["#1f77b4"]

    # Group cases
    rows = []
    for case, parts in zip(cases, parts_norm):
        group = parts[hue_part]
        shade_key = parts[shade_part] if shade_part is not None else ""
        rows.append((str(group), str(shade_key), str(case)))

    groups = sorted(set(r[0] for r in rows))
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    case_colors: Dict[str, str] = {}
    for g in groups:
        group_rows = [r for r in rows if r[0] == g]
        group_rows_sorted = sorted(group_rows, key=lambda r: (r[1], r[2]))
        k = len(group_rows_sorted)
        # Spread shades from darker to lighter.
        positions = np.linspace(-1.0, 1.0, k) if k > 1 else np.array([0.0])
        for (row, pos) in zip(group_rows_sorted, positions):
            _group, _shade_key, case = row
            case_colors[case] = _shade_hex(group_color[g], float(pos))

    return case_colors


@st.cache_data(show_spinner=False)
def cached_clustered_case_colors(cases: Tuple[str, ...], hue_part_override: int) -> Dict[str, str]:
    # hue_part_override: -1 => auto; otherwise 0-based case-part index.
    return build_clustered_case_colors(list(cases), None if int(hue_part_override) < 0 else int(hue_part_override))


def _estimate_legend_height_px(n_traces: int, width_px: int, legend_entrywidth: int) -> int:
    usable_w = max(1, int(width_px) - int(LEFT_MARGIN_PX) - int(RIGHT_MARGIN_PX))
    cols = max(1, int(usable_w // max(1, int(legend_entrywidth))))
    rows = int(np.ceil(float(n_traces) / float(cols))) if n_traces > 0 else 0
    row_h = int(np.ceil(float(STYLE["legend_font_size_px"]) * float(LEGEND_ROW_HEIGHT_FACTOR)))
    return int(rows) * int(row_h) + int(LEGEND_PADDING_PX)


# ---- Data loading ----
@st.cache_data(show_spinner=False)
def load_fs_sweep_xlsx(path_or_buf) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(path_or_buf)
    for name in ["R1", "X1", "R0", "X0"]:
        if name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
            # Frequency column normalization
            freq_col = None
            for c in df.columns:
                c_norm = str(c).strip().lower().replace(" ", "")
                if c_norm in ["frequency(hz)", "frequencyhz", "frequency_"]:
                    freq_col = c
                    break
                if str(c).strip().lower() in ["frequency (hz)", "frequency"]:
                    freq_col = c
                    break
            if freq_col is None:
                if "Frequency (Hz)" in df.columns:
                    freq_col = "Frequency (Hz)"
                else:
                    raise ValueError(f"Sheet '{name}' missing 'Frequency (Hz)' column")
            df = df.rename(columns={freq_col: "Frequency (Hz)"})
            df["Frequency (Hz)"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
            df = df.dropna(subset=["Frequency (Hz)"])
            value_cols = [c for c in df.columns if c != "Frequency (Hz)"]
            if value_cols:
                df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

            # Prepare numpy arrays once per file load; reused during reruns to speed trace building.
            freq_hz = df["Frequency (Hz)"].to_numpy(copy=False)
            series_map: Dict[object, np.ndarray] = {}
            for c in df.columns:
                if c == "Frequency (Hz)":
                    continue
                series_map[c] = df[c].to_numpy(copy=False)
            df.attrs["__prepared_arrays__"] = (freq_hz, series_map)
            dfs[name] = df
    return dfs


def list_case_columns(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    return [c for c in df.columns if c != "Frequency (Hz)"]


def split_case_location(name: str) -> Tuple[str, Optional[str]]:
    if "__" in str(name):
        base, loc = str(name).split("__", 1)
        loc = loc if loc else None
        return base, loc
    return str(name), None


def display_case_name(name: str) -> str:
    base, _ = split_case_location(name)
    return base


def prepare_sheet_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    cached = getattr(df, "attrs", {}).get("__prepared_arrays__")
    if cached is not None:
        return cached

    freq_hz = df["Frequency (Hz)"].to_numpy(copy=False)
    series_map: Dict[object, np.ndarray] = {}
    for c in df.columns:
        if c == "Frequency (Hz)":
            continue
        series_map[c] = df[c].to_numpy(copy=False)
    return freq_hz, series_map


@st.cache_data(show_spinner=False)
def split_case_parts(cases: List[str]) -> Tuple[List[List[str]], List[str]]:
    if not cases:
        return [], []
    temp_parts: List[Tuple[List[str], str]] = []
    max_parts = 0
    for name in cases:
        base_name, location = split_case_location(name)
        base_parts = str(base_name).split("_")
        max_parts = max(max_parts, len(base_parts))
        temp_parts.append((base_parts, location or ""))

    normalized: List[List[str]] = []
    for base_parts, location in temp_parts:
        padded = list(base_parts)
        if len(padded) < max_parts:
            padded.extend([""] * (max_parts - len(padded)))
        padded.append(location or "")
        normalized.append(padded)

    labels = [f"Case part {i+1}" for i in range(max_parts)] + ["Location"]
    return normalized, labels


@st.cache_data(show_spinner=False)
def build_js_case_metadata(cases: Tuple[str, ...]) -> Tuple[List[Dict[str, object]], List[str]]:
    if not cases:
        return [], []
    parts_matrix, part_labels = split_case_parts(list(cases))
    if not part_labels:
        return [], []

    labels_no_loc = list(part_labels[:-1]) if part_labels[-1] == "Location" else list(part_labels)
    part_width = len(labels_no_loc)
    out: List[Dict[str, object]] = []
    for case, row in zip(cases, parts_matrix):
        parts = [str(v) for v in row[:part_width]]
        out.append(
            {
                "case_id": str(case),
                "display_case": str(display_case_name(case)),
                "parts": parts,
            }
        )
    return out, labels_no_loc


@st.cache_data(show_spinner=False)
def _infer_auto_hue_part_label(cases: Tuple[str, ...], part_count: int) -> str:
    # Mirrors Auto hue-part choice used by build_clustered_case_colors(..., hue_part_override=None).
    if not cases or int(part_count) <= 0:
        return ""

    bases = [split_case_location(c)[0] for c in cases]
    split_parts = [str(b).split("_") for b in bases]
    max_parts = max((len(p) for p in split_parts), default=0)
    if max_parts <= 0:
        return ""

    parts_norm = [p + [""] * (max_parts - len(p)) for p in split_parts]
    uniq_counts = [len(set(row[i] for row in parts_norm)) for i in range(max_parts)]
    varying = [i for i, n in enumerate(uniq_counts) if n > 1]
    hue_part = 0 if not varying else sorted(varying, key=lambda i: (-uniq_counts[i], i))[0]
    hue_part = max(0, min(int(part_count) - 1, int(hue_part)))
    return f"Case part {int(hue_part) + 1}"


@st.cache_data(show_spinner=False)
def build_js_color_maps(cases: Tuple[str, ...], part_count: int) -> Tuple[List[str], Dict[str, Dict[str, str]], str]:
    options = ["Auto"] + [f"Case part {i}" for i in range(1, int(part_count) + 1)]
    color_maps: Dict[str, Dict[str, str]] = {}
    for idx, label in enumerate(options):
        hue_idx = -1 if idx == 0 else idx - 1
        cmap = cached_clustered_case_colors(cases, int(hue_idx))
        color_maps[label] = {str(k): str(v) for k, v in cmap.items()}
    auto_color_part_label = _infer_auto_hue_part_label(cases, int(part_count))
    return options, color_maps, auto_color_part_label


def list_location_values(cases: List[str]) -> List[str]:
    vals = sorted({str(split_case_location(c)[1] or "") for c in cases})
    return vals if vals else [""]


def filter_cases_by_location(cases: List[str], location_value: str) -> List[str]:
    loc = str(location_value or "")
    return [c for c in cases if str(split_case_location(c)[1] or "") == loc]


def compute_common_n_range(f_series: List[pd.Series], f_base: float) -> Tuple[float, float]:
    vals: List[float] = []
    for s in f_series:
        if s is None:
            continue
        v = s
        if not pd.api.types.is_numeric_dtype(v):
            v = pd.to_numeric(v, errors="coerce")
        v = v.dropna()
        if not v.empty:
            vals.extend([v.min() / f_base, v.max() / f_base])
    if not vals:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return (0.0, 1.0) if (not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi) else (lo, hi)


def make_spline_traces(
    df: pd.DataFrame,
    cases: List[str],
    f_base: float,
    y_title: str,
    smooth: float,
    enable_spline: bool,
    strip_location_suffix: bool,
    case_colors: Dict[str, str],
) -> Tuple[List[BaseTraceType], Optional[pd.Series]]:
    if df is None:
        return [], None
    cd, y_map = prepare_sheet_arrays(df)
    n = cd / float(f_base)
    traces: List[BaseTraceType] = []
    TraceCls = go.Scatter if enable_spline else go.Scattergl
    for case in cases:
        y = y_map.get(case)
        if y is None:
            continue
        color = str(case_colors.get(case, "#1f77b4"))
        line_cfg = dict(color=color)
        tr = TraceCls(
            x=n,
            y=y,
            customdata=cd,
            mode="lines",
            name=display_case_name(case) if strip_location_suffix else str(case),
            meta={
                "kind": "line",
                "case_id": str(case),
                "display_case": str(display_case_name(case)),
                "legend_color": color,
            },
            line=line_cfg,
            opacity=1.0,
            showlegend=True,
            hovertemplate=(
                "Case=%{fullData.name}<br>f=%{customdata:.1f} Hz" + f"<br>{y_title}=%{{y}}<extra></extra>"
            ),
        )
        if enable_spline and isinstance(tr, go.Scatter):
            spline_line = dict(
                shape="spline",
                smoothing=float(smooth),
                simplify=False,
                color=color,
            )
            tr.update(line=spline_line)
        traces.append(tr)
    return traces, df["Frequency (Hz)"]


def apply_common_layout(
    fig: go.Figure,
    plot_height: int,
    y_title: str,
    legend_entrywidth: int,
    n_traces: int,
    use_auto_width: bool,
    figure_width_px: int,
):
    font_base = dict(family=STYLE["font_family"], color=STYLE["font_color"])
    # Reserve legend space below the plot so the legend stays at the bottom on-page.
    # Increase x-axis reserved space when fonts are larger to reduce title/tick overlap on zoom.
    bottom_axis_px = int(
        max(
            int(BOTTOM_AXIS_PX),
            int(
                round(
                    float(STYLE["tick_font_size_px"]) * float(BOTTOM_AXIS_TICK_MULT)
                    + float(STYLE["axis_title_font_size_px"]) * float(BOTTOM_AXIS_TITLE_MULT)
                )
            ),
        )
    )
    est_width_px = int(figure_width_px) if not use_auto_width else int(AUTO_WIDTH_ESTIMATE_PX)
    legend_h_full = _estimate_legend_height_px(int(n_traces), est_width_px, int(legend_entrywidth))
    legend_h = min(int(WEB_LEGEND_MAX_HEIGHT_PX), int(legend_h_full) + int(WEB_LEGEND_EXTRA_PAD_PX))
    total_height = int(plot_height) + int(TOP_MARGIN_PX) + int(bottom_axis_px) + int(legend_h)
    legend_y = -float(bottom_axis_px) / float(max(1, int(plot_height)))

    # Y-axis overlap fix: keep bottom legend behavior, but grow left margin with font sizes
    # so y tick labels and y title don't collide after zoom.
    left_margin_px = int(
        max(
            int(LEFT_MARGIN_PX),
            int(
                round(
                    float(STYLE["tick_font_size_px"]) * float(LEFT_MARGIN_TICK_MULT)
                    + float(STYLE["axis_title_font_size_px"]) * float(LEFT_MARGIN_TITLE_MULT)
                )
            ),
        )
    )

    fig.update_layout(
        autosize=bool(use_auto_width),
        height=total_height,
        # Keep zoom/pan on Streamlit reruns (including when case list changes).
        uirevision="keep",
        font=dict(
            **font_base,
            size=int(STYLE["base_font_size_px"]),
        ),
        margin=dict(
            l=left_margin_px,
            r=RIGHT_MARGIN_PX,
            t=TOP_MARGIN_PX,
            b=int(bottom_axis_px) + int(legend_h),
        ),
        margin_autoexpand=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            entrywidth=int(legend_entrywidth),
            entrywidthmode="pixels",
            font=dict(**font_base, size=int(STYLE["legend_font_size_px"])),
        ),
    )
    if not use_auto_width:
        fig.update_layout(width=int(figure_width_px), autosize=False)

    x_title = "Harmonic number n = f / f_base"
    y_title_txt = str(y_title)
    if bool(STYLE.get("bold_axis_titles", True)):
        x_title = f"<b>{x_title}</b>"
        y_title_txt = f"<b>{y_title_txt}</b>"

    axis_title_font = dict(**font_base, size=int(STYLE["axis_title_font_size_px"]))
    tick_font = dict(**font_base, size=int(STYLE["tick_font_size_px"]))

    x_title_standoff = STYLE.get("xaxis_title_standoff_px")
    if x_title_standoff is None:
        x_title_standoff = int(
            max(
                int(AXIS_TITLE_STANDOFF_MIN_PX),
                round(float(STYLE["tick_font_size_px"]) * float(AXIS_TITLE_STANDOFF_TICK_MULT)),
            )
        )
    else:
        x_title_standoff = int(x_title_standoff)

    y_title_standoff = STYLE.get("yaxis_title_standoff_px")
    if y_title_standoff is None:
        y_title_standoff = int(
            max(
                int(AXIS_TITLE_STANDOFF_MIN_PX),
                round(float(STYLE["tick_font_size_px"]) * float(AXIS_TITLE_STANDOFF_TICK_MULT)),
            )
        )
    else:
        y_title_standoff = int(y_title_standoff)
    fig.update_xaxes(
        title_text=x_title,
        tick0=1,
        dtick=1,
        title_font=axis_title_font,
        tickfont=tick_font,
        automargin=True,
        title_standoff=x_title_standoff,
    )
    fig.update_yaxes(
        title_text=y_title_txt,
        title_font=axis_title_font,
        tickfont=tick_font,
        automargin=True,
        title_standoff=y_title_standoff,
    )


def build_plot_spline(df: Optional[pd.DataFrame], cases: List[str], f_base: float, plot_height: int, y_title: str,
                      smooth: float, enable_spline: bool, legend_entrywidth: int, strip_location_suffix: bool,
                      use_auto_width: bool, figure_width_px: int, case_colors: Dict[str, str],
                      ) -> Tuple[go.Figure, Optional[pd.Series]]:
    traces, f_series = make_spline_traces(
        df,
        cases,
        f_base,
        y_title,
        smooth,
        enable_spline,
        strip_location_suffix,
        case_colors,
    )
    fig = go.Figure(data=traces)
    legend_n = sum(1 for t in traces if bool(getattr(t, "showlegend", True)))
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, legend_n, use_auto_width, figure_width_px)
    return fig, f_series


def build_x_over_r_spline(df_r: Optional[pd.DataFrame], df_x: Optional[pd.DataFrame], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_entrywidth: int,
                          enable_spline: bool,
                          strip_location_suffix: bool, use_auto_width: bool, figure_width_px: int,
                          case_colors: Dict[str, str],
                          ) -> Tuple[go.Figure, Optional[pd.Series], int, int]:
    xr_dropped = 0
    xr_total = 0
    f_series = None
    eps = float(XR_EPS)
    TraceCls = go.Scatter if enable_spline else go.Scattergl
    traces: List[BaseTraceType] = []
    if df_r is not None and df_x is not None:
        cd, r_map = prepare_sheet_arrays(df_r)
        _cd2, x_map = prepare_sheet_arrays(df_x)
        n = cd / float(f_base)
        both = [c for c in cases if (c in r_map and c in x_map)]
        f_series = df_r["Frequency (Hz)"]
        for case in both:
            r = r_map[case]
            x = x_map[case]
            denom_ok = np.abs(r) >= eps
            bad = (~denom_ok) | np.isnan(r) | np.isnan(x)
            y = np.where(denom_ok, x / r, np.nan)
            xr_dropped += int(np.count_nonzero(bad))
            xr_total += int(r.size)
            color = str(case_colors.get(case, "#1f77b4"))
            line_cfg = dict(color=color)
            tr = TraceCls(
                x=n,
                y=y,
                customdata=cd,
                mode="lines",
                name=display_case_name(case) if strip_location_suffix else str(case),
                meta={
                    "kind": "line",
                    "case_id": str(case),
                    "display_case": str(display_case_name(case)),
                    "legend_color": color,
                },
                line=line_cfg,
                opacity=1.0,
                showlegend=True,
                hovertemplate=(
                    "Case=%{fullData.name}<br>f=%{customdata:.1f} Hz<br>X/R=%{y}<extra></extra>"
                ),
            )
            if enable_spline and isinstance(tr, go.Scatter):
                spline_line = dict(
                    shape="spline",
                    smoothing=float(smooth),
                    simplify=False,
                    color=color,
                )
                tr.update(line=spline_line)
            traces.append(tr)
    fig = go.Figure(data=traces)
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    legend_n = sum(1 for t in traces if bool(getattr(t, "showlegend", True)))
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, legend_n, use_auto_width, figure_width_px)
    return fig, f_series, xr_dropped, xr_total


def build_rx_scatter_animated(
    df_r: Optional[pd.DataFrame],
    df_x: Optional[pd.DataFrame],
    cases: List[str],
    seq_label: str,
    case_colors: Dict[str, str],
    plot_height: int,
    axis_cases: Optional[List[str]] = None,
) -> Tuple[go.Figure, int]:
    fig = go.Figure()
    if df_r is None or df_x is None or not cases:
        fig.update_layout(height=500)
        return fig, 0

    fr, r_map = prepare_sheet_arrays(df_r)
    fx, x_map = prepare_sheet_arrays(df_x)
    if fr.size == 0 or fx.size == 0:
        fig.update_layout(height=500)
        return fig, 0

    freq_candidates = sorted(
        {
            float(v)
            for v in np.concatenate([fr[np.isfinite(fr)], fx[np.isfinite(fx)]], axis=0)
            if np.isfinite(v)
        }
    )
    if not freq_candidates:
        fig.update_layout(height=500)
        return fig, 0
    init_idx = int(min(len(freq_candidates) - 1, max(0, len(freq_candidates) // 2)))
    freq_candidates_arr = np.asarray(freq_candidates, dtype=float)

    r_global_min: Optional[float] = None
    r_global_max: Optional[float] = None
    x_global_min: Optional[float] = None
    x_global_max: Optional[float] = None
    case_arrays: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for case in cases:
        r_arr = r_map.get(case)
        x_arr = x_map.get(case)
        if r_arr is None or x_arr is None:
            continue
        case_arrays.append((str(case), r_arr, x_arr))

    axis_case_list = list(axis_cases) if axis_cases is not None else list(cases)
    for case in axis_case_list:
        r_arr = r_map.get(case)
        x_arr = x_map.get(case)
        if r_arr is None or x_arr is None:
            continue
        r_finite = r_arr[np.isfinite(r_arr)]
        x_finite = x_arr[np.isfinite(x_arr)]
        if r_finite.size > 0:
            r_min_i = float(np.min(r_finite))
            r_max_i = float(np.max(r_finite))
            r_global_min = r_min_i if r_global_min is None else min(r_global_min, r_min_i)
            r_global_max = r_max_i if r_global_max is None else max(r_global_max, r_max_i)
        if x_finite.size > 0:
            x_min_i = float(np.min(x_finite))
            x_max_i = float(np.max(x_finite))
            x_global_min = x_min_i if x_global_min is None else min(x_global_min, x_min_i)
            x_global_max = x_max_i if x_global_max is None else max(x_global_max, x_max_i)

    # Precompute nearest R/X row indices for each slider frequency once.
    idx_r_for_freq = np.array([int(np.argmin(np.abs(fr - float(f_sel)))) for f_sel in freq_candidates_arr], dtype=int)
    idx_x_for_freq = np.array([int(np.argmin(np.abs(fx - float(f_sel)))) for f_sel in freq_candidates_arr], dtype=int)

    def frame_data_for_freq_idx(fi: int) -> Tuple[dict, int]:
        idx_r = int(idx_r_for_freq[int(fi)])
        idx_x = int(idx_x_for_freq[int(fi)])
        f_used_r = float(fr[idx_r])
        f_used_x = float(fx[idx_x])
        f_used = 0.5 * (f_used_r + f_used_x)

        xs: List[float] = []
        ys: List[float] = []
        cds: List[List[object]] = []
        colors: List[str] = []
        ids: List[str] = []

        for case, r_arr, x_arr in case_arrays:
            if idx_r >= int(r_arr.size) or idx_x >= int(x_arr.size):
                continue
            r_v = r_arr[idx_r]
            x_v = x_arr[idx_x]
            if not np.isfinite(r_v) or not np.isfinite(x_v):
                continue
            xs.append(float(r_v))
            ys.append(float(x_v))
            cds.append([str(case), str(display_case_name(case)), float(f_used)])
            colors.append(str(case_colors.get(case, "#1f77b4")))
            ids.append(str(case))

        trace = dict(
            type="scatter",
            x=xs,
            y=ys,
            mode="markers",
            name="Cases",
            customdata=cds,
            ids=ids,
            hovertemplate="Case=%{customdata[1]}<br>f=%{customdata[2]:.1f} Hz<br>R=%{x}<br>X=%{y}<extra></extra>",
            marker=dict(
                color=colors,
                size=float(SELECTED_MARKER_SIZE),
                opacity=1.0,
                line=dict(width=0),
            ),
            showlegend=False,
            meta={"kind": "points"},
        )
        return trace, len(xs)

    f0 = float(freq_candidates[init_idx])
    tr0, _ = frame_data_for_freq_idx(init_idx)
    fig.add_trace(go.Scatter(**tr0))

    frames: List[go.Frame] = []
    for i, f_sel in enumerate(freq_candidates):
        tr_i, _ = frame_data_for_freq_idx(i)
        frames.append(
            go.Frame(
                name=f"{float(f_sel):.6g}",
                data=[go.Scatter(**tr_i)],
                traces=[0],
                layout=go.Layout(title=f"R vs X at f ~ {float(f_sel):.1f} Hz ({seq_label})"),
            )
        )
    fig.frames = frames
    slider_steps = [
        dict(
            method="animate",
            args=[
                [f"{float(f_sel):.6g}"],
                dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0)),
            ],
            label=f"{float(f_sel):.1f}",
        )
        for f_sel in freq_candidates
    ]

    shapes: List[dict] = [
        dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.45)", width=1)),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="rgba(0,0,0,0.45)", width=1)),
    ]
    fig.update_layout(
        title=f"R vs X at f ~ {f0:.1f} Hz ({seq_label})",
        xaxis_title=f"R{1 if seq_label == 'Positive' else 0} (Ohm)",
        yaxis_title=f"X{1 if seq_label == 'Positive' else 0} (Ohm)",
        height=max(420, int(round(float(plot_height) * float(RX_SCATTER_HEIGHT_FACTOR)))),
        dragmode="zoom",
        # Keep click handling deterministic in custom JS stager.
        # Using "event" avoids Plotly's built-in selection-state side effects.
        clickmode="event",
        shapes=shapes,
        sliders=[
            dict(
                active=int(init_idx),
                currentvalue=dict(prefix="Frequency (Hz): "),
                pad=dict(t=20),
                steps=slider_steps,
            )
        ],
    )

    if (
        r_global_min is not None and r_global_max is not None and np.isfinite(r_global_min) and np.isfinite(r_global_max)
        and x_global_min is not None and x_global_max is not None and np.isfinite(x_global_min) and np.isfinite(x_global_max)
    ):
        rx_pad = max(1e-12, 0.04 * max(1e-12, float(r_global_max - r_global_min)))
        xx_pad = max(1e-12, 0.04 * max(1e-12, float(x_global_max - x_global_min)))
        fig.update_xaxes(range=[float(r_global_min - rx_pad), float(r_global_max + rx_pad)])
        fig.update_yaxes(range=[float(x_global_min - xx_pad), float(x_global_max + xx_pad)])
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig, len(freq_candidates)


def _make_plot_item(
    kind: str,
    fig: go.Figure,
    f_ref: Optional[pd.Series],
    filename: str,
    button_label: str,
    chart_key: str,
) -> Dict[str, object]:
    return {
        "kind": str(kind),
        "fig": fig,
        "f_ref": f_ref,
        "filename": str(filename),
        "button_label": str(button_label),
        "chart_key": str(chart_key),
    }


def _render_client_png_download(
    filename: str,
    scale: int,
    button_label: str,
    plot_height: int,
    legend_entrywidth: int,
    plot_index: int,
):
    dom_id = hashlib.sha1(f"{filename}|{scale}|{plot_height}|{legend_entrywidth}|{plot_index}".encode("utf-8")).hexdigest()[
        : int(EXPORT_DOM_ID_HASH_LEN)
    ]
    html = f"""
    <div id="exp-{dom_id}">
      <button id="btn-{dom_id}" style="padding:6px 10px; font-size: 0.9rem; cursor:pointer;">
        {button_label}
      </button>
      <div id="plot-{dom_id}" style="width:1px; height:1px; position:absolute; left:-99999px; top:-99999px;"></div>
    </div>
    <script>
      const scale = {int(scale)};
      const plotHeight = {int(plot_height)};
      const topMargin = {int(TOP_MARGIN_PX)};
      const bottomAxis = {int(BOTTOM_AXIS_PX)};
      const legendPad = {int(LEGEND_PADDING_PX)};
      const legendEntryWidth = {int(legend_entrywidth)};
      const plotIndex = {int(plot_index)};
      const filename = {json.dumps(filename)};
      const fallbackLegendFontSize = {int(STYLE["legend_font_size_px"])};

      async function doExport() {{
        try {{
          const Plotly = window.parent?.Plotly;
          if (!Plotly) return;
          const plots = window.parent?.document?.querySelectorAll?.("div.js-plotly-plot");
          if (!plots || plots.length <= plotIndex) return;
          const gd = plots[plotIndex];
          if (!gd) return;

          const r = gd.getBoundingClientRect();
          const widthPx = Math.floor(r.width || 0);
          if (!widthPx) return;

          const legendFontSize =
            gd?._fullLayout?.legend?.font?.size ||
            gd?._fullLayout?.font?.size ||
            fallbackLegendFontSize;
          // Legend row height: keep tight to avoid bottom whitespace.
          const legendRowH = Math.ceil(legendFontSize * {float(EXPORT_LEGEND_ROW_HEIGHT_FACTOR)});
      const legendFontFamily = {json.dumps(STYLE["font_family"])};
      const legendFontColor = {json.dumps(STYLE["font_color"])};

      const leftMarginBase = {int(LEFT_MARGIN_PX)};
      const rightMargin = {int(RIGHT_MARGIN_PX)};
      const tickFontSize = {int(STYLE["tick_font_size_px"])};
      const axisTitleFontSize = {int(STYLE["axis_title_font_size_px"])};
      const leftMarginPx = Math.max(leftMarginBase, Math.round(tickFontSize * {float(LEFT_MARGIN_TICK_MULT)} + axisTitleFontSize * {float(LEFT_MARGIN_TITLE_MULT)}));

          const data = Array.isArray(gd.data) ? gd.data : [];
          const data2 = data.map((tr) => {{
            const t = Object.assign({{}}, tr);
            if (t.type === "scattergl") t.type = "scatter";
            return t;
          }});
          const legendItems = [];
          for (const tr of data2) {{
            if (tr && tr.showlegend === false) continue;
            const name = tr && tr.name ? String(tr.name) : "";
            if (!name) continue;
            // Prefer the actual trace styling (so export always matches on-page plot).
            const color =
              (tr.line && tr.line.color) ? tr.line.color :
              (tr.marker && tr.marker.color) ? tr.marker.color :
              (tr.meta && tr.meta.legend_color) ? tr.meta.legend_color :
              {json.dumps(EXPORT_FALLBACK_COLOR)};
            legendItems.push({{name, color}});
          }}

          const usableW = Math.max(1, widthPx - leftMarginPx - rightMargin);

          // Estimate needed entry width using canvas text measurement to avoid overlap for long names.
          let maxTextW = 0;
          try {{
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (ctx) {{
              ctx.font = legendFontSize + "px " + legendFontFamily;
              for (const it of legendItems) {{
                const w = ctx.measureText(it.name).width || 0;
                if (w > maxTextW) maxTextW = w;
              }}
            }}
          }} catch (e) {{}}

          const sampleLinePx = Math.max({int(EXPORT_SAMPLE_LINE_MIN_PX)}, Math.round({float(EXPORT_SAMPLE_LINE_MULT)} * legendFontSize));
          const sampleGapPx = Math.max({int(EXPORT_SAMPLE_GAP_MIN_PX)}, Math.round({float(EXPORT_SAMPLE_GAP_MULT)} * legendFontSize));
          const textPadPx = Math.max({int(EXPORT_TEXT_PAD_MIN_PX)}, Math.round({float(EXPORT_TEXT_PAD_MULT)} * legendFontSize));
          const neededEntryPx = Math.ceil(sampleLinePx + sampleGapPx + maxTextW + textPadPx);
          const entryPx = Math.max(1, Math.max(legendEntryWidth, neededEntryPx));

          const cols = Math.max(1, Math.floor(usableW / entryPx));
          const rows = Math.ceil(legendItems.length / cols);
          // Total legend area in bottom margin.
          // Add a small tail so the last row doesn't look cramped, but avoid large blank space.
          const legendH = (rows * legendRowH) + legendPad + Math.ceil({float(EXPORT_LEGEND_TAIL_FONT_MULT)} * legendFontSize);

          const newHeight = plotHeight + topMargin + bottomAxis + legendH;
          const newMarginB = bottomAxis + legendH;

          const container = document.getElementById("plot-{dom_id}");
          if (!container) return;
          container.style.width = widthPx + "px";
          container.style.height = newHeight + "px";

          const baseLayout = Object.assign({{}}, gd.layout || {{}});
          baseLayout.width = widthPx;
          baseLayout.height = newHeight;
          baseLayout.autosize = false;
          baseLayout.margin = Object.assign({{}}, baseLayout.margin || {{}});
          baseLayout.margin.t = topMargin;
          baseLayout.margin.l = leftMarginPx;
          baseLayout.margin.r = rightMargin;
          baseLayout.margin.b = newMarginB;
          // Disable Plotly legend and draw a manual legend in the bottom margin so it never scrolls/clips.
          baseLayout.showlegend = false;
          for (const tr of data2) {{
            tr.showlegend = false;
          }}

          const ann = Array.isArray(baseLayout.annotations) ? baseLayout.annotations.slice() : [];
          const shp = Array.isArray(baseLayout.shapes) ? baseLayout.shapes.slice() : [];

          // Spread columns across the available width to avoid side whitespace and tight columns.
          const colW = usableW / cols;
          const xPadPx = Math.max(0, Math.min({int(EXPORT_COL_PADDING_MAX_PX)}, Math.floor(colW * {float(EXPORT_COL_PADDING_FRAC)})));

          for (let i = 0; i < legendItems.length; i++) {{
            const row = Math.floor(i / cols);
            const col = i % cols;
            const x0 = (col * colW + xPadPx) / usableW;
            const x1 = x0 + (sampleLinePx / usableW);
            const y = -(bottomAxis + legendPad + (row + {float(EXPORT_LEGEND_ROW_Y_OFFSET)}) * legendRowH) / Math.max(1, plotHeight);

            shp.push({{
              type: "line",
              xref: "paper",
              yref: "paper",
              x0, x1,
              y0: y, y1: y,
              line: {{color: legendItems[i].color, width: 2}}
            }});

            ann.push({{
              xref: "paper",
              yref: "paper",
              x: x1 + (sampleGapPx / usableW),
              y,
              xanchor: "left",
              yanchor: "middle",
              showarrow: false,
              align: "left",
              text: legendItems[i].name,
              font: {{size: legendFontSize, family: legendFontFamily, color: legendFontColor}}
            }});
          }}

          baseLayout.annotations = ann;
          baseLayout.shapes = shp;

          await Plotly.newPlot(container, data2, baseLayout, {{displayModeBar: false, staticPlot: true}});
          const url = await Plotly.toImage(container, {{format: "png", width: widthPx, height: newHeight, scale}});
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
        }} catch (e) {{
        }} finally {{
          try {{
            const container = document.getElementById("plot-{dom_id}");
            if (container) {{
              container.innerHTML = "";
              container.style.width = "1px";
              container.style.height = "1px";
            }}
          }} catch (e) {{}}
        }}
      }}

      document.getElementById("btn-{dom_id}").addEventListener("click", doExport);
    </script>
    """
    components.html(html, height=70)


def _render_rx_client_step_buttons(plot_index: int, data_id: str, chart_id: str) -> None:
    dom_id = hashlib.sha1(f"rx-step:{int(plot_index)}:{str(data_id)}:{str(chart_id)}".encode("utf-8")).hexdigest()[: int(EXPORT_DOM_ID_HASH_LEN)]
    html = f"""
    <style>
      #rx-step-{dom_id} {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin: 2px 0 6px 0;
        font-family: "Open Sans", verdana, arial, sans-serif;
      }}
      #rx-step-{dom_id} .rx-btn {{
        padding: 4px 10px;
        font-size: 12px;
        cursor: pointer;
        font-family: inherit;
        color: #222;
      }}
      #rx-step-{dom_id} .rx-showonly {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #222;
      }}
    </style>
    <div id="rx-step-{dom_id}">
      <button id="rx-prev-{dom_id}" type="button" class="rx-btn">&#8592; Prev frequency</button>
      <button id="rx-next-{dom_id}" type="button" class="rx-btn">Next frequency &#8594;</button>
      <button id="rx-clear-{dom_id}" type="button" class="rx-btn">Clear list</button>
      <button id="rx-csv-{dom_id}" type="button" class="rx-btn">Download selected CSV</button>
      <label class="rx-showonly">
        <input id="rx-showonly-{dom_id}" type="checkbox" />
        <span>Show only selected sweeps</span>
      </label>
    </div>
    <script>
    (function() {{
      const root = document.getElementById("rx-step-{dom_id}");
      if (!root || root.__bound) return;
      root.__bound = true;
      const dataId = {json.dumps(str(data_id))};
      const chartId = {json.dumps(str(chart_id))};
      const stateKey = String(dataId) + "|" + String(chartId);

      function getPlots() {{
        const out = [];
        try {{
          const doc = window.parent.document;
          const blocks = doc.querySelectorAll('div[data-testid="stPlotlyChart"], div.stPlotlyChart');
          for (const block of blocks) {{
            const fr = block.querySelector("iframe");
            if (fr && fr.contentWindow && fr.contentWindow.document) {{
              const gd = fr.contentWindow.document.querySelector("div.js-plotly-plot");
              if (gd) {{
                out.push(gd);
                continue;
              }}
            }}
            const gd2 = block.querySelector("div.js-plotly-plot");
            if (gd2) out.push(gd2);
          }}
        }} catch (e) {{}}
        return out;
      }}

      function getSelectionApi() {{
        try {{
          const rootWin = window.parent;
          const apiStore = rootWin && rootWin.__fsCaseUiApi && typeof rootWin.__fsCaseUiApi === "object" ? rootWin.__fsCaseUiApi : null;
          if (!apiStore) return null;
          const api = apiStore[stateKey];
          return api && typeof api === "object" ? api : null;
        }} catch (e) {{}}
        return null;
      }}

      function syncSelectionControls() {{
        try {{
          const api = getSelectionApi();
          if (!api || typeof api.getState !== "function") return;
          const st = api.getState();
          const cb = document.getElementById("rx-showonly-{dom_id}");
          if (cb && st && Object.prototype.hasOwnProperty.call(st, "showOnlySelected")) {{
            cb.checked = Boolean(st.showOnlySelected);
          }}
        }} catch (e) {{}}
      }}

      function step(delta) {{
        try {{
          const plots = getPlots();
          const gd = plots[{int(plot_index)}];
          if (!gd) return;
          const frames =
            (gd._transitionData && Array.isArray(gd._transitionData._frames) && gd._transitionData._frames) ||
            (Array.isArray(gd.frames) ? gd.frames : []);
          if (!Array.isArray(frames) || frames.length === 0) return;

          let active = 0;
          try {{
            const sliders = gd.layout && Array.isArray(gd.layout.sliders) ? gd.layout.sliders : [];
            if (sliders.length && sliders[0] && sliders[0].active != null) active = Number(sliders[0].active);
          }} catch (e) {{}}
          if (!Number.isFinite(active)) active = 0;
          active = Math.max(0, Math.min(frames.length - 1, Math.floor(active)));
          const next = Math.max(0, Math.min(frames.length - 1, active + Number(delta || 0)));
          const frameObj = frames[next];
          const frameName = frameObj && frameObj.name != null ? String(frameObj.name) : String(next);
          const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
          const Plotly = win && win.Plotly ? win.Plotly : null;
          if (!Plotly || !Plotly.animate) return;
          Plotly.animate(gd, [frameName], {{
            mode: "immediate",
            frame: {{ duration: 0, redraw: false }},
            transition: {{ duration: 0 }},
          }});
        }} catch (e) {{}}
      }}

      const prev = document.getElementById("rx-prev-{dom_id}");
      const next = document.getElementById("rx-next-{dom_id}");
      const showOnly = document.getElementById("rx-showonly-{dom_id}");
      const clearBtn = document.getElementById("rx-clear-{dom_id}");
      const csvBtn = document.getElementById("rx-csv-{dom_id}");
      if (prev) prev.addEventListener("click", function(ev) {{ ev.preventDefault(); step(-1); }});
      if (next) next.addEventListener("click", function(ev) {{ ev.preventDefault(); step(1); }});
      if (showOnly) {{
        showOnly.addEventListener("change", function() {{
          try {{
            const api = getSelectionApi();
            if (api && typeof api.setShowOnlySelected === "function") {{
              api.setShowOnlySelected(Boolean(showOnly.checked));
            }}
          }} catch (e) {{}}
        }});
      }}
      if (clearBtn) {{
        clearBtn.addEventListener("click", function(ev) {{
          ev.preventDefault();
          try {{
            const api = getSelectionApi();
            if (api && typeof api.clearSelection === "function") {{
              api.clearSelection();
              syncSelectionControls();
            }}
          }} catch (e) {{}}
        }});
      }}
      if (csvBtn) {{
        csvBtn.addEventListener("click", function(ev) {{
          ev.preventDefault();
          try {{
            const api = getSelectionApi();
            if (api && typeof api.downloadSelectedCsv === "function") {{
              api.downloadSelectedCsv();
            }}
          }} catch (e) {{}}
        }});
      }}
      const timerKey = "__fsRxToolbarSyncTimer_{dom_id}";
      try {{
        if (window[timerKey]) clearInterval(window[timerKey]);
        window[timerKey] = setInterval(syncSelectionControls, 250);
      }} catch (e) {{}}
      syncSelectionControls();
    }})();
    </script>
    """
    components.html(html, height=86)


def main():
    st.title("FS Sweep Visualizer (Spline)")

    # Data source
    default_path = "FS_sweep.xlsx"
    st.sidebar.header("Data Source")
    up = st.sidebar.file_uploader(
        "Upload Excel",
        type=["xlsx"],
        key="xlsx_uploader",
        on_change=_note_upload_change,
        help="If empty, loads 'FS_sweep.xlsx' from this folder.",
    )
    st.sidebar.markdown("---")
    data_id = "unknown"
    try:
        if up is not None:
            data = load_fs_sweep_xlsx(up)
            try:
                cached = st.session_state.get("uploaded_file_sha1_10")
                data_id = str(cached) if cached else hashlib.sha1(up.getvalue()).hexdigest()[:10]
            except Exception:
                data_id = f"upload:{getattr(up, 'name', 'file')}"
        elif os.path.exists(default_path):
            data = load_fs_sweep_xlsx(default_path)
            try:
                data_id = f"local:{int(os.path.getmtime(default_path))}"
            except Exception:
                data_id = "local"
            st.sidebar.info(f"Loaded local file: {default_path}")
        else:
            st.warning("Upload an Excel file or place 'FS_sweep.xlsx' here.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.stop()

    upload_nonce = int(st.session_state.get("upload_nonce", 0))

    seq_key = "seq_label_control"
    base_key = "base_freq_control"
    if st.session_state.get(seq_key) not in ("Positive", "Zero"):
        st.session_state[seq_key] = "Positive"
    if st.session_state.get(base_key) not in ("50 Hz", "60 Hz"):
        st.session_state[base_key] = "50 Hz"

    # Global controls (Streamlit-side; these rerun by design).
    st.sidebar.header("Controls")
    seq_label = str(st.session_state.get(seq_key, "Positive"))
    seq = ("R1", "X1") if seq_label == "Positive" else ("R0", "X0")
    base_label = str(st.session_state.get(base_key, "50 Hz"))
    f_base = 50.0 if base_label.startswith("50") else 60.0
    plot_height = st.sidebar.slider("Plot area height (px)", min_value=100, max_value=1000, value=400, step=25)
    use_auto_width = st.sidebar.checkbox("Auto width (fit container)", value=True)
    figure_width_px = DEFAULT_FIGURE_WIDTH_PX
    if not use_auto_width:
        figure_width_px = st.sidebar.slider("Figure width (px)", min_value=800, max_value=2200, value=DEFAULT_FIGURE_WIDTH_PX, step=50)

    enable_spline = st.sidebar.checkbox("Spline (slow)", value=False)
    spline_selection_reset_key = "selection_reset_nonce:spline_toggle"
    spline_prev_state_key = "selection_reset_prev_spline"
    prev_spline_state = st.session_state.get(spline_prev_state_key, None)
    if prev_spline_state is None:
        st.session_state[spline_prev_state_key] = bool(enable_spline)
    elif bool(prev_spline_state) != bool(enable_spline):
        st.session_state[spline_selection_reset_key] = int(st.session_state.get(spline_selection_reset_key, 0)) + 1
        st.session_state[spline_prev_state_key] = bool(enable_spline)
    selection_reset_token = int(st.session_state.get(spline_selection_reset_key, 0))

    smooth = float(DEFAULT_SPLINE_SMOOTHING)
    if enable_spline:
        prev_smooth = st.session_state.get("spline_smoothing", float(DEFAULT_SPLINE_SMOOTHING))
        try:
            prev_smooth_f = float(prev_smooth)
        except Exception:
            prev_smooth_f = float(DEFAULT_SPLINE_SMOOTHING)
        prev_smooth_f = max(float(SPLINE_SMOOTHING_MIN), min(float(SPLINE_SMOOTHING_MAX), prev_smooth_f))
        smooth = st.sidebar.slider(
            "Spline smoothing",
            min_value=float(SPLINE_SMOOTHING_MIN),
            max_value=float(SPLINE_SMOOTHING_MAX),
            value=prev_smooth_f,
            step=float(SPLINE_SMOOTHING_STEP),
            key="spline_smoothing",
        )
    st.sidebar.markdown("---")

    # Prepare sequence sheets and full case list.
    df_r = data.get(seq[0])
    df_x = data.get(seq[1])
    if df_r is None and df_x is None:
        st.error(f"Missing sheets for sequence '{seq_label}' ({seq[0]}/{seq[1]}).")
        st.stop()

    all_cases = sorted(list({*list_case_columns(df_r), *list_case_columns(df_x)}))
    if not all_cases:
        st.warning("No case columns found in the selected sequence sheets.")
        st.stop()

    location_values = list_location_values(all_cases)
    location_labels = [("<empty>" if str(v) == "" else str(v)) for v in location_values]
    location_label_to_value = {lbl: val for lbl, val in zip(location_labels, location_values)}
    location_key = f"location_select:{data_id}:{seq_label}"
    if location_key not in st.session_state or st.session_state.get(location_key) not in location_labels:
        st.session_state[location_key] = location_labels[0]
    # Show-plots controls (R vs X default on).
    st.sidebar.header("Show plots")
    show_plot_rx = st.sidebar.checkbox("R vs X scatter", value=True)
    show_plot_x = st.sidebar.checkbox("X", value=True)
    show_plot_r = st.sidebar.checkbox("R", value=False)
    show_plot_xr = st.sidebar.checkbox("X/R", value=False)
    if not (show_plot_x or show_plot_r or show_plot_xr or show_plot_rx):
        st.warning("Select at least one plot to display.")
        st.stop()
    st.sidebar.markdown("---")

    # Legend/Export controls
    st.sidebar.header("Legend & Export")
    auto_legend_entrywidth = st.sidebar.checkbox("Auto legend column width", value=True)
    legend_entrywidth = 180
    if not auto_legend_entrywidth:
        legend_entrywidth = st.sidebar.slider("Legend column width (px)", min_value=50, max_value=300, value=180, step=10)
    download_area = st.sidebar.container()
    st.sidebar.markdown("---")

    st.sidebar.header("Case Filters & Selection")
    st.sidebar.radio("Sequence", ["Positive", "Zero"], key=seq_key)
    st.sidebar.radio("Base frequency", ["50 Hz", "60 Hz"], key=base_key)
    selected_location_label = st.sidebar.radio("Location", options=location_labels, key=location_key)
    selected_location = str(location_label_to_value.get(str(selected_location_label), ""))
    interactive_controls_area = st.sidebar.container()
    st.sidebar.markdown("---")

    # Validate required sheets for enabled plots.
    if (show_plot_r or show_plot_xr or show_plot_rx) and df_r is None:
        st.error(f"Sheet '{seq[0]}' is missing, but R, X/R and/or R vs X scatter is enabled.")
        st.stop()
    if (show_plot_x or show_plot_xr or show_plot_rx) and df_x is None:
        st.error(f"Sheet '{seq[1]}' is missing, but X, X/R and/or R vs X scatter is enabled.")
        st.stop()

    location_cases = filter_cases_by_location(all_cases, selected_location)
    if not location_cases:
        st.warning("No cases found for the selected location.")
        st.stop()

    cases_tuple = tuple(location_cases)
    cases_meta, part_labels = build_js_case_metadata(cases_tuple)
    color_by_options, color_maps, auto_color_part_label = build_js_color_maps(cases_tuple, len(part_labels))
    default_color_map = dict(color_maps.get("Auto", {}))
    case_colors_line = {c: str(default_color_map.get(c, "#1f77b4")) for c in location_cases}
    case_colors_scatter = {c: str(default_color_map.get(c, "#1f77b4")) for c in location_cases}

    # Legend/Export controls
    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "scale": int(EXPORT_IMAGE_SCALE),
        }
    }

    # Build all cases for selected location on server-side.
    # Case-part filtering and selection styling are applied client-side in JS.
    cases_for_line = list(location_cases)
    strip_location_suffix = True

    if auto_legend_entrywidth:
        legend_cases = list(cases_for_line)
        display_names = [display_case_name(c) for c in legend_cases]
        max_len = max((len(n) for n in display_names), default=12)
        legend_font_px = int(STYLE["legend_font_size_px"])
        approx_char_px = max(7, int(round(0.60 * float(legend_font_px))))
        base_px = max(44, int(round(3.5 * float(legend_font_px))))  # symbol + padding inside a legend item

        # Only used to cap export legend columns; use the configured width when available.
        est_width_px = int(figure_width_px)
        usable_w = max(1, int(est_width_px) - int(LEFT_MARGIN_PX) - int(RIGHT_MARGIN_PX))
        desired = int(max_len) * int(approx_char_px) + int(base_px)
        legend_entrywidth = _clamp_int(desired, 50, min(900, usable_w))
        if legend_entrywidth >= int(usable_w * 0.95):
            legend_entrywidth = usable_w

    # Client-side zoom persistence.
    plot_order: List[str] = []
    if show_plot_rx:
        plot_order.append("rx")
    if show_plot_x:
        plot_order.append("x")
    if show_plot_r:
        plot_order.append("r")
    if show_plot_xr:
        plot_order.append("xr")

    bind_nonce_key = f"zoom_bind_nonce:{data_id}"
    st.session_state[bind_nonce_key] = int(st.session_state.get(bind_nonce_key, 0)) + 1

    plotly_relayout_listener(
        data_id=data_id,
        plot_count=len(plot_order),
        plot_ids=plot_order,
        debounce_ms=150,
        nonce=int(st.session_state.get(bind_nonce_key, 0)),
        reset_token=int(upload_nonce),
    )

    # Build plots
    r_title = "R1 (\u03A9)" if seq_label == "Positive" else "R0 (\u03A9)"
    x_title = "X1 (\u03A9)" if seq_label == "Positive" else "X0 (\u03A9)"
    plot_items: List[Dict[str, object]] = []
    xr_dropped = 0
    xr_total = 0

    if show_plot_x:
        x_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:x"
        x_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:x"
        x_sig_payload = {
            "kind": "x",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(x_title),
        }
        x_sig = hashlib.sha1(json.dumps(x_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(x_cache_sig_key) != x_sig or (x_cache_fig_key not in st.session_state):
            fig_x_built, _ = build_plot_spline(
                df_x,
                cases_for_line,
                f_base,
                plot_height,
                x_title,
                smooth,
                enable_spline,
                legend_entrywidth,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[x_cache_sig_key] = x_sig
            st.session_state[x_cache_fig_key] = fig_x_built.to_dict()
        fig_x = go.Figure(st.session_state.get(x_cache_fig_key, {}))
        f_x = df_x["Frequency (Hz)"] if df_x is not None else None
        plot_items.append(_make_plot_item("x", fig_x, f_x, "X_full_legend.png", "X\nPNG", "plot_x"))

    if show_plot_r:
        r_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:r"
        r_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:r"
        r_sig_payload = {
            "kind": "r",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(r_title),
        }
        r_sig = hashlib.sha1(json.dumps(r_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(r_cache_sig_key) != r_sig or (r_cache_fig_key not in st.session_state):
            fig_r_built, _ = build_plot_spline(
                df_r,
                cases_for_line,
                f_base,
                plot_height,
                r_title,
                smooth,
                enable_spline,
                legend_entrywidth,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[r_cache_sig_key] = r_sig
            st.session_state[r_cache_fig_key] = fig_r_built.to_dict()
        fig_r = go.Figure(st.session_state.get(r_cache_fig_key, {}))
        f_r = df_r["Frequency (Hz)"] if df_r is not None else None
        plot_items.append(_make_plot_item("r", fig_r, f_r, "R_full_legend.png", "R\nPNG", "plot_r"))

    if show_plot_xr:
        xr_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:xr"
        xr_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:xr"
        xr_cache_meta_key = f"line_fig_meta:{data_id}:{seq_label}:xr"
        xr_sig_payload = {
            "kind": "xr",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(seq_label),
        }
        xr_sig = hashlib.sha1(json.dumps(xr_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(xr_cache_sig_key) != xr_sig or (xr_cache_fig_key not in st.session_state):
            fig_xr_built, _, xr_dropped_built, xr_total_built = build_x_over_r_spline(
                df_r,
                df_x,
                cases_for_line,
                f_base,
                plot_height,
                seq_label,
                smooth,
                legend_entrywidth,
                enable_spline,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[xr_cache_sig_key] = xr_sig
            st.session_state[xr_cache_fig_key] = fig_xr_built.to_dict()
            st.session_state[xr_cache_meta_key] = {
                "xr_dropped": int(xr_dropped_built),
                "xr_total": int(xr_total_built),
            }
        fig_xr = go.Figure(st.session_state.get(xr_cache_fig_key, {}))
        xr_meta = st.session_state.get(xr_cache_meta_key, {}) if isinstance(st.session_state.get(xr_cache_meta_key, {}), dict) else {}
        xr_dropped = int(xr_meta.get("xr_dropped", 0))
        xr_total = int(xr_meta.get("xr_total", 0))
        f_xr = df_r["Frequency (Hz)"] if df_r is not None else None
        plot_items.append(_make_plot_item("xr", fig_xr, f_xr, "X_over_R_full_legend.png", "X/R\nPNG", "plot_xr"))

    f_refs = [it["f_ref"] for it in plot_items if it.get("f_ref") is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    for it in plot_items:
        fig = it["fig"]
        if isinstance(fig, go.Figure):
            fig.update_xaxes(range=[n_lo, n_hi])

    # Render
    location_caption = selected_location if selected_location else "<empty>"
    st.subheader(f"Sequence: {seq_label} | Base: {int(f_base)} Hz | Location: {location_caption}")
    if show_plot_xr and xr_total > 0 and xr_dropped > 0:
        st.caption(f"X/R: dropped {xr_dropped} of {xr_total} points where |R| < {XR_EPS_DISPLAY} or data missing.")

    export_scale = int(EXPORT_IMAGE_SCALE)
    with download_area:
        st.subheader("Download (Full Legend)")
        if plot_items:
            st.caption("Browser PNG download (temporarily expands the on-page chart legend, then downloads).")
            cols = st.columns(len(plot_items))
            line_plot_base_index = 1 if show_plot_rx else 0
            for idx, it in enumerate(plot_items):
                with cols[idx]:
                    _render_client_png_download(
                        filename=str(it["filename"]),
                        scale=export_scale,
                        button_label=str(it["button_label"]),
                        plot_height=plot_height,
                        legend_entrywidth=legend_entrywidth,
                        plot_index=int(line_plot_base_index + idx),
                    )
        else:
            st.caption("No X, R, or X/R line plots selected for full-legend download.")

    scatter_slot = st.container()
    line_slot = st.container()

    with line_slot:
        for idx, it in enumerate(plot_items):
            fig = it["fig"]
            chart_key = str(it["chart_key"])
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=bool(use_auto_width), config=download_config, key=chart_key)
            if idx < len(plot_items) - 1:
                st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    rx_status_dom_id = ""
    rx_freq_steps_for_bridge = 0
    with scatter_slot:
        if show_plot_rx:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.subheader("R vs X Scatter")

            rx_filter_sig_key = f"rx_filter_sig:{data_id}:{seq_label}"
            rx_fig_sig_key = f"rx_fig_sig:{data_id}:{seq_label}"
            rx_fig_cache_key = f"rx_fig_cache:{data_id}:{seq_label}"
            rx_fig_steps_key = f"rx_fig_steps:{data_id}:{seq_label}"

            filter_sig = hashlib.sha1("|".join(sorted(location_cases)).encode("utf-8")).hexdigest()[:12]
            prev_filter_sig = str(st.session_state.get(rx_filter_sig_key, ""))
            if prev_filter_sig != filter_sig:
                st.session_state[rx_filter_sig_key] = filter_sig
                st.session_state.pop(rx_fig_sig_key, None)
                st.session_state.pop(rx_fig_cache_key, None)
                st.session_state.pop(rx_fig_steps_key, None)

            # Location-based baseline for scatter axis limits.
            location_cases_for_axes = list(location_cases)

            rx_sig_payload = {
                "seq": str(seq_label),
                "plot_h": int(plot_height),
                "cases": list(location_cases),
                "axis_cases": list(location_cases_for_axes),
                "colors": [[str(c), str(case_colors_scatter.get(c, "#1f77b4"))] for c in location_cases],
            }
            rx_sig = hashlib.sha1(json.dumps(rx_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

            if st.session_state.get(rx_fig_sig_key) != rx_sig or (rx_fig_cache_key not in st.session_state):
                rx_fig_built, rx_steps_built = build_rx_scatter_animated(
                    df_r=df_r,
                    df_x=df_x,
                    cases=list(location_cases),
                    seq_label=seq_label,
                    case_colors=case_colors_scatter,
                    plot_height=int(plot_height),
                    axis_cases=list(location_cases_for_axes),
                )
                st.session_state[rx_fig_sig_key] = rx_sig
                st.session_state[rx_fig_cache_key] = rx_fig_built.to_dict()
                st.session_state[rx_fig_steps_key] = int(rx_steps_built)

            rx_fig = go.Figure(st.session_state.get(rx_fig_cache_key, {}))
            rx_fig.update_layout(uirevision=f"rx:{seq_label}")
            rx_freq_steps = int(st.session_state.get(rx_fig_steps_key, 0))
            rx_freq_steps_for_bridge = int(rx_freq_steps)
            rx_status_dom_id = f"rx-status-{hashlib.sha1(f'{data_id}:{seq_label}:{selected_location}'.encode('utf-8')).hexdigest()[:10]}"

            st.plotly_chart(rx_fig, use_container_width=bool(use_auto_width), config=download_config, key="plot_rx")
            rx_plot_index = int(plot_order.index("rx")) if "rx" in plot_order else 0
            _render_rx_client_step_buttons(rx_plot_index, data_id=data_id, chart_id=f"{seq_label}:{selected_location}")
            st.markdown(
                (
                    f"<div id=\"{rx_status_dom_id}\" style=\"font-size:0.92rem; color:#666; margin:2px 0 2px 0;\">"
                    f"R vs X points shown: {len(location_cases)} | Frequency steps: {int(rx_freq_steps)}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.caption("Point clicks toggle selection. Case-part/color/selection controls are in the sidebar.")

    sel_bind_nonce_key = f"selection_bind_nonce:{data_id}:{seq_label}:{selected_location}"
    st.session_state[sel_bind_nonce_key] = int(st.session_state.get(sel_bind_nonce_key, 0)) + 1

    with interactive_controls_area:
        plotly_selection_bridge(
            data_id=data_id,
            chart_id=f"{seq_label}:{selected_location}",
            plot_ids=list(plot_order),
            cases_meta=list(cases_meta),
            part_labels=list(part_labels),
            color_by_options=list(color_by_options),
            color_maps=dict(color_maps),
            auto_color_part_label=str(auto_color_part_label),
            color_by_default="Auto",
            show_only_default=False,
            selected_marker_size=float(SELECTED_MARKER_SIZE),
            dim_marker_opacity=float(DIM_MARKER_OPACITY),
            selected_line_width=float(SELECTED_LINE_WIDTH),
            dim_line_width=float(DIM_LINE_WIDTH),
            dim_line_opacity=float(DIM_LINE_OPACITY),
            dim_line_color=str(DIM_LINE_COLOR),
            f_base=float(f_base),
            n_min=float(n_lo),
            n_max=float(n_hi),
            show_harmonics_default=True,
            bin_width_hz_default=0.0,
            rx_status_dom_id=str(rx_status_dom_id),
            rx_freq_steps=int(rx_freq_steps_for_bridge),
            reset_token=int(upload_nonce),
            selection_reset_token=int(selection_reset_token),
            render_nonce=int(st.session_state.get(sel_bind_nonce_key, 0)),
            enable_selection=bool(show_plot_rx),
        )


if __name__ == "__main__":
    main()
