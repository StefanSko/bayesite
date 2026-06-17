use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt;

use nuts_rs::rand::rngs::StdRng;
use nuts_rs::rand::SeedableRng;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
};

#[derive(Debug, Clone)]
struct TargetSpec {
    name: &'static str,
    mean: Vec<f64>,
    precision: Vec<Vec<f64>>,
}

impl TargetSpec {
    fn dim(&self) -> usize {
        self.mean.len()
    }
}

#[derive(Debug, Clone)]
struct GaussianLogp {
    mean: Vec<f64>,
    precision: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct GaussianLogpError;

impl fmt::Display for GaussianLogpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Gaussian log density failed")
    }
}

impl Error for GaussianLogpError {}

impl LogpError for GaussianLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl HasDims for GaussianLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([(
            "unconstrained_parameter".to_string(),
            self.mean.len() as u64,
        )])
    }
}

impl CpuLogpFunc for GaussianLogp {
    type LogpError = GaussianLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let dim = self.mean.len();
        let mut diff = vec![0.0; dim];
        for i in 0..dim {
            diff[i] = position[i] - self.mean[i];
        }
        let mut precision_diff = vec![0.0; dim];
        for (row, value) in precision_diff.iter_mut().enumerate() {
            *value = self.precision[row]
                .iter()
                .zip(&diff)
                .map(|(p, d)| p * d)
                .sum();
        }
        let quadratic: f64 = diff.iter().zip(&precision_diff).map(|(d, pd)| d * pd).sum();
        for i in 0..dim {
            grad[i] = -precision_diff[i];
        }
        Ok(-0.5 * quadratic)
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

#[derive(Debug)]
struct Args {
    targets: Vec<String>,
    seed: u64,
    chains: usize,
    warmup: u64,
    draws: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            targets: all_target_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            seed: 20240621,
            chains: 4,
            warmup: 500,
            draws: 1000,
        }
    }
}

fn usage_error(message: &str) -> Box<dyn Error> {
    format!(
        "{message}; usage: nuts-rs-runner [--targets comma,separated] [--seed N] [--chains N] [--warmup N] [--draws N]"
    )
    .into()
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| usage_error(&format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--targets" => {
                args.targets = value
                    .split(',')
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect();
                if args.targets.is_empty() {
                    return Err(usage_error("--targets must name at least one target"));
                }
            }
            "--seed" => args.seed = value.parse()?,
            "--chains" => args.chains = value.parse()?,
            "--warmup" => args.warmup = value.parse()?,
            "--draws" => args.draws = value.parse()?,
            _ => return Err(usage_error(&format!("unknown flag {flag}"))),
        }
    }
    if args.chains == 0 {
        return Err(usage_error("--chains must be at least 1"));
    }
    if args.draws == 0 {
        return Err(usage_error("--draws must be at least 1"));
    }
    Ok(args)
}

fn all_target_names() -> [&'static str; 4] {
    [
        "scalar_standard",
        "shifted_scaled",
        "vector_diagonal",
        "correlated_mvn",
    ]
}

fn target_spec(name: &str) -> Option<TargetSpec> {
    match name {
        "scalar_standard" => Some(TargetSpec {
            name: "scalar_standard",
            mean: vec![0.0],
            precision: vec![vec![1.0]],
        }),
        "shifted_scaled" => Some(TargetSpec {
            name: "shifted_scaled",
            mean: vec![2.0],
            precision: vec![vec![1.0 / 9.0]],
        }),
        "vector_diagonal" => Some(TargetSpec {
            name: "vector_diagonal",
            mean: vec![0.0, 2.0, -1.0],
            precision: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 4.0, 0.0],
                vec![0.0, 0.0, 0.25],
            ],
        }),
        "correlated_mvn" => Some(TargetSpec {
            name: "correlated_mvn",
            mean: vec![1.0, -0.5],
            // Inverse of covariance [[1.0, 0.6], [0.6, 1.0]].
            precision: vec![vec![1.5625, -0.9375], vec![-0.9375, 1.5625]],
        }),
        _ => None,
    }
}

struct ChainOutput {
    divergences: usize,
    draws: Vec<Vec<f64>>,
}

struct TargetOutput {
    name: &'static str,
    chains: Vec<ChainOutput>,
}

fn sample_target(spec: &TargetSpec, args: &Args) -> Result<TargetOutput, Box<dyn Error>> {
    let settings = DiagNutsSettings {
        num_tune: args.warmup,
        num_draws: args.draws,
        seed: args.seed,
        num_chains: args.chains,
        ..Default::default()
    };
    let mut chains = Vec::with_capacity(args.chains);
    for chain_id in 0..args.chains {
        let chain_seed = args
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(chain_id as u64 + 1));
        let mut rng = StdRng::seed_from_u64(chain_seed);
        let math = CpuMath::new(GaussianLogp {
            mean: spec.mean.clone(),
            precision: spec.precision.clone(),
        });
        let mut chain = settings.new_chain(chain_id as u64, math, &mut rng);
        chain.set_position(&vec![0.1; spec.dim()])?;
        let mut draws = Vec::with_capacity(args.draws as usize);
        let mut divergences = 0usize;
        while draws.len() < args.draws as usize {
            let (draw, progress) = chain.draw()?;
            if !progress.tuning {
                if progress.diverging {
                    divergences += 1;
                }
                draws.push(draw.to_vec());
            }
        }
        chains.push(ChainOutput { divergences, draws });
    }
    Ok(TargetOutput {
        name: spec.name,
        chains,
    })
}

fn push_f64(out: &mut String, value: f64) -> Result<(), Box<dyn Error>> {
    if !value.is_finite() {
        return Err(format!("non-finite draw value {value}").into());
    }
    out.push_str(&format!("{value:.17}"));
    Ok(())
}

fn push_output_json(out: &mut String, target: &TargetOutput) -> Result<(), Box<dyn Error>> {
    out.push_str("{\"name\":\"");
    out.push_str(target.name);
    out.push_str("\",\"chains\":[");
    for (chain_index, chain) in target.chains.iter().enumerate() {
        if chain_index > 0 {
            out.push(',');
        }
        out.push_str("{\"chain\":");
        out.push_str(&chain_index.to_string());
        out.push_str(",\"divergences\":");
        out.push_str(&chain.divergences.to_string());
        out.push_str(",\"draws\":[");
        for (draw_index, draw) in chain.draws.iter().enumerate() {
            if draw_index > 0 {
                out.push(',');
            }
            out.push('[');
            for (dim, value) in draw.iter().enumerate() {
                if dim > 0 {
                    out.push(',');
                }
                push_f64(out, *value)?;
            }
            out.push(']');
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let mut outputs = Vec::with_capacity(args.targets.len());
    for name in &args.targets {
        let spec = target_spec(name).ok_or_else(|| {
            format!(
                "unknown target {name:?}; supported targets are {:?}",
                all_target_names()
            )
        })?;
        outputs.push(sample_target(&spec, &args)?);
    }

    let mut json = String::new();
    json.push_str("{\"runner\":\"nuts-rs\",\"targets\":[");
    for (index, output) in outputs.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        push_output_json(&mut json, output)?;
    }
    json.push_str("]}");
    println!("{json}");
    Ok(())
}
