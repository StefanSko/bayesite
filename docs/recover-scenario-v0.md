# `recover_scenario` schema (v0-provisional)

Input document for `bayesite recover --scenario <scenario.json|->`. One JSON
object; unknown or duplicate fields are rejected.

```json
{
  "recover_scenario": "v0-provisional",
  "data": {"n": 2, "x": {"dtype": "float64", "shape": [2], "values": [0.0, 0.1]}},
  "seed": 7,
  "interval": 0.8,
  "sample": {"chains": 4, "warmup": 1000, "draws": 1000, "max_treedepth": 10, "target_accept": 0.8}
}
```

## Fields

| Field | Required | Type | Constraints |
|---|---|---|---|
| `recover_scenario` | yes | string | must be exactly `"v0-provisional"` (format marker) |
| `data` | yes | object | data document keyed by data variable name, same shape as the `--data` input of `sample` |
| `seed` | yes | integer | `0..=9223372036854775807` |
| `interval` | no | number | central posterior interval probability in `(0, 1)`; default `0.8` |
| `sample` | no | object | sampler settings, see below |

## The `sample` object

Accepted keys (unknown keys are rejected):

| Key | Type | Constraints | Default |
|---|---|---|---|
| `chains` | integer | at least 1 | 4 |
| `warmup` | integer | at least 0 | 1000 |
| `draws` | integer | at least 4 (reports include diagnostics) | 1000 |
| `max_treedepth` | integer | `1..=20` | 10 |
| `target_accept` | number | in `(0, 1)` | 0.8 |

## Output

`recover` writes one JSON report with `recover_format: "v0-provisional"` and
`workflow_format: "v0-provisional"`; see [artifacts-v0.md](artifacts-v0.md).
