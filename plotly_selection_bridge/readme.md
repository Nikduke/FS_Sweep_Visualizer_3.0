# plotly_selection_bridge

Client-side control bridge for case filtering, selection, color styling, harmonics overlays, and scatter status updates.

## Files

- `index.html`
- `listener.js`

## Input args

- `data_id` (string)
- `chart_id` (string)
- `plot_ids` (string[]): visible plot order as rendered in Streamlit (current default starts with `rx`, then line plots)
- `cases_meta` (object[]):
  - `case_id` (string)
  - `display_case` (string)
  - `parts` (string[])
- `part_labels` (string[])
- `color_by_options` (string[])
- `color_maps` (object): `{option -> {case_id -> color_hex}}`
- `auto_color_part_label` (string): case-part label used when `Color=Auto` (for filter color dots)
- `color_by_default` (string)
- `show_only_default` (bool)
- `selected_marker_size` (float)
- `dim_marker_opacity` (float)
- `selected_line_width` (float)
- `dim_line_width` (float)
- `dim_line_opacity` (float)
- `dim_line_color` (string)
- `f_base` (float)
- `n_min` (float)
- `n_max` (float)
- `show_harmonics_default` (bool)
- `bin_width_hz_default` (float)
- `rx_status_dom_id` (string): optional parent DOM id for scatter status text
- `rx_freq_steps` (int): fallback step count for status text
- `reset_token` (int)
- `render_nonce` (int)
- `enable_selection` (bool)

## Behavior

1. Renders control panel UI inside component iframe.
2. Computes allowed cases from case-part filters.
3. Applies selection style layer:
   - dim mode (default)
   - hide mode (`Show only selected sweeps`)
4. Restyles line plots (`x/r/xr`) without rerun:
   - `visible`, `showlegend`, line color/width/opacity
   - when selection exists, legend entries are limited to selected cases
5. Applies harmonics overlays on line plots from JS controls.
6. Restyles scatter points (`rx`) without rerun.
7. Updates scatter status text in parent DOM (`rx_status_dom_id`) with case-filtered visible count.
8. Handles scatter click selection toggle.
9. Supports selection-table actions (clear/remove/import/csv).
10. Reapplies scatter styling on frequency animation events (`plotly_sliderchange`, `plotly_animated`) and frame-object refreshes to prevent selection flicker.
11. In case-part filters, shows color dots to the right of values for the active `Color` grouping.
    - if `Color=Auto`, uses `auto_color_part_label`.

## State model

- stored in `window.parent.__fsCaseUiStore`
- key: `{data_id}|{chart_id}`
- persists across normal reruns in same page session
- resets on `reset_token` change

## Python roundtrip policy

- Normal interactions do not emit `streamlit:setComponentValue`.
- Updates are applied directly to existing Plotly DOM.
