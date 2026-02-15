# plotly_relayout_listener

Client-side zoom persistence bridge for Plotly charts rendered by Streamlit.

## Files

- `index.html`
- `listener.js`

## Input args

- `data_id` (string): dataset identity
- `plot_count` (int): number of charts to bind
- `plot_ids` (string[]): chart IDs in render order from Streamlit (e.g. current default starts with `rx`)
- `debounce_ms` (int): write debounce for localStorage
- `nonce` (int): rerender/rebind trigger
- `reset_token` (int): reset signal for current dataset
- `bind_tries` (int)
- `bind_interval_ms` (int)
- `ignore_autorange_ms` (int)

## Behavior

1. Finds Streamlit Plotly DOM nodes.
2. Binds `plotly_relayout` listeners by plot index.
3. Persists ranges to localStorage key:
   - `fsSweepZoom:{data_id}:{plot_id}`
4. On `reset_token` change, clears keys for current `data_id`.
5. Reapplies stored ranges via `Plotly.relayout` on bind.
6. Ignores initial autorange-only relayout events during mount window.

## Notes

- No zoom payload is sent back to Python.
- Persistence is browser-local only.
