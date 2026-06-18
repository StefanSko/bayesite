//! Shared helpers for v0-provisional workflow artifact metadata.

use crate::error::{Error, ErrorKind};
use crate::json::Value;

pub(crate) const V0_PROVISIONAL: &str = "v0-provisional";
pub(crate) const CHAIN_INDEX_BASE: &str = "zero_based_chain_id";
pub(crate) const WORKFLOW_FORMAT: &str = V0_PROVISIONAL;
pub(crate) const PARAMETER_SUMMARY_SCALE: &str = "constrained_parameter_value";
pub(crate) const POSTERIOR_DRAW_INDEX_BASE: &str = "zero_based_retained_draw_order";
pub(crate) const PRIOR_PREDICTIVE_DRAW_INDEX_BASE: &str = "zero_based_prior_predictive_draw_order";
pub(crate) const REPLICATE_INDEX_BASE: &str = "zero_based_replicate_order";
pub(crate) const SIMULATION_INDEX_BASE: &str = "zero_based_simulation_order";
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

pub(crate) const PRIOR_PREDICTIVE_DRAWS: ArtifactIdentity = ArtifactIdentity {
    kind: "prior_predictive_draws",
    scope: "declared_data_conditioned_site_draws",
};

pub(crate) const POSTERIOR_PREDICTIVE_DRAWS: ArtifactIdentity = ArtifactIdentity {
    kind: "posterior_predictive_draws",
    scope: "observed_data_conditioned_replicated_observed_data_draws",
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

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

pub(crate) fn report_count_value(count: usize, context: &str) -> Result<Value, Error> {
    if count > i64::MAX as usize {
        Err(invalid(format!(
            "{context} must be in 0..=9223372036854775807 because artifacts report counts as JSON integers"
        )))
    } else {
        Ok(Value::Int(count as i64))
    }
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
        assert_eq!(
            artifact_identity_entries(PRIOR_PREDICTIVE_DRAWS),
            vec![
                (
                    "artifact_kind".to_string(),
                    Value::Str("prior_predictive_draws".to_string())
                ),
                (
                    "artifact_scope".to_string(),
                    Value::Str("declared_data_conditioned_site_draws".to_string())
                ),
            ]
        );
    }

    #[test]
    fn prior_predictive_draw_index_base_names_its_ordering() {
        assert_eq!(
            PRIOR_PREDICTIVE_DRAW_INDEX_BASE,
            "zero_based_prior_predictive_draw_order"
        );
    }

    #[test]
    fn workflow_report_labels_name_their_scales_and_indexes() {
        assert_eq!(WORKFLOW_FORMAT, V0_PROVISIONAL);
        assert_eq!(PARAMETER_SUMMARY_SCALE, "constrained_parameter_value");
        assert_eq!(REPLICATE_INDEX_BASE, "zero_based_replicate_order");
        assert_eq!(SIMULATION_INDEX_BASE, "zero_based_simulation_order");
    }

    #[test]
    fn report_count_value_rejects_unreportable_counts() {
        assert_eq!(report_count_value(7, "test count").unwrap(), Value::Int(7));
        assert_eq!(
            report_count_value(i64::MAX as usize + 1, "test count")
                .unwrap_err()
                .message,
            "test count must be in 0..=9223372036854775807 because artifacts report counts as JSON integers"
        );
    }
}
