//! Shared helpers for v0-provisional workflow artifact metadata.

use crate::json::Value;

pub(crate) const V0_PROVISIONAL: &str = "v0-provisional";
pub(crate) const CHAIN_INDEX_BASE: &str = "zero_based_chain_id";
pub(crate) const POSTERIOR_DRAW_INDEX_BASE: &str = "zero_based_retained_draw_order";
pub(crate) const RHAT_STATISTIC: &str = "split_rhat";
pub(crate) const ESS_STATISTIC: &str = "effective_sample_size_geyer_initial_monotone_sequence";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArtifactIdentity {
    pub(crate) kind: &'static str,
    pub(crate) scope: &'static str,
}

pub(crate) const POSTERIOR_DRAWS: ArtifactIdentity = ArtifactIdentity {
    kind: "posterior_draws",
    scope: "observed_data_conditioned_parameter_draws",
};

pub(crate) fn format_marker_field(name: &str) -> (String, Value) {
    (name.to_string(), Value::Str(V0_PROVISIONAL.to_string()))
}

pub(crate) fn artifact_identity_entries(identity: ArtifactIdentity) -> Vec<(String, Value)> {
    vec![
        (
            "artifact_kind".to_string(),
            Value::Str(identity.kind.to_string()),
        ),
        (
            "artifact_scope".to_string(),
            Value::Str(identity.scope.to_string()),
        ),
    ]
}

pub(crate) fn shape_value(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|&dim| Value::Int(dim as i64)).collect())
}

pub(crate) fn coordinate_order_value(shape: &[usize]) -> Value {
    if shape.contains(&0) {
        return Value::Array(Vec::new());
    }
    let size = shape.iter().product::<usize>().max(1);
    Value::Array(
        (0..size)
            .map(|flat| {
                let mut remainder = flat;
                let mut coordinate = vec![0usize; shape.len()];
                for axis in (0..shape.len()).rev() {
                    let dim = shape[axis];
                    coordinate[axis] = remainder % dim;
                    remainder /= dim;
                }
                Value::Array(
                    coordinate
                        .iter()
                        .map(|&index| Value::Int(index as i64))
                        .collect(),
                )
            })
            .collect(),
    )
}

pub(crate) fn entry_order_value<T>(entries: &[(String, T)]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|(name, _)| Value::Str(name.clone()))
            .collect(),
    )
}

pub(crate) fn u64_order_value(values: &[u64]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|&value| Value::Int(value as i64))
            .collect(),
    )
}

pub(crate) fn i64_order_value(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|&value| Value::Int(value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_coord(entries: &[&[i64]]) -> Value {
        Value::Array(
            entries
                .iter()
                .map(|coord| Value::Array(coord.iter().map(|&index| Value::Int(index)).collect()))
                .collect(),
        )
    }

    #[test]
    fn coordinate_order_is_zero_based_row_major() {
        assert_eq!(coordinate_order_value(&[]), int_coord(&[&[]]));
        assert_eq!(coordinate_order_value(&[3]), int_coord(&[&[0], &[1], &[2]]));
        assert_eq!(
            coordinate_order_value(&[2, 2]),
            int_coord(&[&[0, 0], &[0, 1], &[1, 0], &[1, 1]])
        );
        assert_eq!(coordinate_order_value(&[0]), Value::Array(Vec::new()));
    }

    #[test]
    fn artifact_identity_entries_are_explicit_and_ordered() {
        assert_eq!(
            artifact_identity_entries(POSTERIOR_DRAWS),
            vec![
                (
                    "artifact_kind".to_string(),
                    Value::Str("posterior_draws".to_string())
                ),
                (
                    "artifact_scope".to_string(),
                    Value::Str("observed_data_conditioned_parameter_draws".to_string())
                ),
            ]
        );
    }
}
