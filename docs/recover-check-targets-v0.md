# `recover_check_targets` schema (v0-provisional)

Optional input document for
`bayesite recover-check --targets <targets.json|->`. One JSON object with a
single `targets` field; unknown fields are rejected.

```json
{
  "targets": [
    {"name": "slope", "truth": "beta_true", "posterior": "beta"}
  ]
}
```

## Fields

| Field | Required | Type | Constraints |
|---|---|---|---|
| `targets` | yes | array | at least one target object |

Each target object (unknown fields are rejected):

| Field | Required | Type | Constraints |
|---|---|---|---|
| `name` | yes | string | unique across targets; names the report entry |
| `truth` | yes | string | must name a value in the `--truth` document |
| `posterior` | yes | string | must name a posterior parameter in the fit |

## Default without `--targets`

When `--targets` is omitted, every value in the `--truth` document is
auto-mapped to the posterior parameter of the **same name**
(`name = truth = posterior`). A truth value with no same-named posterior
parameter is an error; pass explicit targets to map differently named
values.

## Output

`recover-check` writes one JSON report with
`recover_check_format: "v0-provisional"`; see
[artifacts-v0.md](artifacts-v0.md).
