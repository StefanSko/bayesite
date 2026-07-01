//! Reverse-mode AD over the closed op set the IR compiles to.
//!
//! A `Tape` is built once per graph: forward values are computed eagerly as
//! ops are pushed, and `backward` walks the tape once to accumulate adjoint
//! tensors. The op set is exactly what the evaluator and the distribution
//! log-densities need — this is not a general autodiff.
//!
//! Because the graph structure of a bound model is fixed (index expressions
//! are parameter-free by IR contract), a built tape can be re-evaluated at a
//! new point without rebuilding it: [`Tape::set_leaf`] updates the input
//! leaves and [`Tape::replay`] re-runs the forward pass in evaluation order
//! through the same `eval_op` code that computed the values at build time.
//! Everything value-dependent — including support masks, which are grad-free
//! predicate ops rather than captured constants — is recomputed on replay.

use crate::error::Error;
use crate::linalg;
use crate::special;
use crate::tensor::{GatherMap, Tensor};

/// Handle to a tape node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(usize);

/// Comparison predicate materialized as a 0.0/1.0 mask tensor.
#[derive(Debug, Clone, Copy)]
enum CmpKind {
    Ge,
    Gt,
    Le,
    Lt,
}

#[derive(Debug, Clone)]
enum Op {
    /// Constant or input leaf; participates in gradients iff marked.
    Leaf,
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    Div(Var, Var),
    Neg(Var),
    Exp(Var),
    Ln(Var),
    /// ln(1 + x)
    Ln1p(Var),
    Sigmoid(Var),
    Softplus(Var),
    Gammaln(Var),
    /// x * ln(y) with xlogy(0, y) = 0
    Xlogy(Var, Var),
    /// Full reduction to a rank-0 scalar.
    Sum(Var),
    /// out[i] = parent[map.map[i]]
    Gather(Var, GatherMap),
    /// Zeros of `len`, with parent segments scattered to fixed positions:
    /// out[positions[i]] = parent[i] for each (parent, positions) pair.
    Scatter {
        len: usize,
        parts: Vec<(Var, Vec<usize>)>,
    },
    /// Elementwise select: cond ? then_v : else_v (all broadcast together).
    Where {
        cond: Var,
        then_v: Var,
        else_v: Var,
    },
    /// Ordered-constraint inverse: x[0]=y[0], x[i]=y[0]+sum_{k<=i} exp(y[k]).
    OrderedInverse(Var),
    /// Elementwise Normal log probability with analytic adjoints.
    NormalLogProb {
        value: Var,
        loc: Var,
        scale: Var,
    },
    /// MultivariateNormal log probability with analytic adjoints.
    MultivariateNormalLogProb {
        value: Var,
        mean: Var,
        scale_tril: Var,
    },
    /// Solve L x = b for lower-triangular L (rank-2) and rank-1 b.
    SolveLower(Var, Var),
    /// Concatenate along the last axis (equal leading dims).
    ConcatLast(Vec<Var>),
    /// Materialized broadcast to the stored shape.
    Broadcast(Var, Vec<usize>),
    Reshape(Var, Vec<usize>),
    /// Grad-free 0.0/1.0 comparison mask with broadcasting.
    Cmp(CmpKind, Var, Var),
    /// Grad-free logical AND of two 0.0/1.0 masks with broadcasting.
    And(Var, Var),
    /// Grad-free elementwise "is an integer" mask.
    IsInteger(Var),
    /// Grad-free scalar mask: 1.0 iff the rank-1 parent strictly increases.
    IsStrictlyIncreasing(Var),
    /// out[row] = base[row * k + clamp(index[row], 0, k-1)] along the last
    /// axis of `base`; the index is re-read per evaluation (grad flows
    /// through `base` only).
    TakeAlongLast {
        base: Var,
        index: Var,
    },
}

struct Node {
    value: Tensor,
    op: Op,
    requires_grad: bool,
    /// Depends (through any op, including grad-free masks) on an input leaf,
    /// so its value must be recomputed on replay.
    dynamic: bool,
}

/// Forward evaluation of one op from its parents' current values. This is
/// the single implementation shared by graph construction and [`Tape::replay`],
/// so a replayed tape is bit-identical to a freshly built one.
fn eval_op(nodes: &[Node], op: &Op) -> Tensor {
    let value = |v: &Var| -> &Tensor { &nodes[v.0].value };
    match op {
        Op::Leaf => unreachable!("leaf values are assigned, not computed"),
        Op::Add(a, b) => value(a)
            .binary(value(b), |x, y| x + y)
            .expect("shapes broadcast"),
        Op::Sub(a, b) => value(a)
            .binary(value(b), |x, y| x - y)
            .expect("shapes broadcast"),
        Op::Mul(a, b) => value(a)
            .binary(value(b), |x, y| x * y)
            .expect("shapes broadcast"),
        Op::Div(a, b) => value(a)
            .binary(value(b), |x, y| x / y)
            .expect("shapes broadcast"),
        Op::Neg(a) => value(a).map(|x| -x),
        Op::Exp(a) => value(a).map(f64::exp),
        Op::Ln(a) => value(a).map(f64::ln),
        Op::Ln1p(a) => value(a).map(f64::ln_1p),
        Op::Sigmoid(a) => value(a).map(special::sigmoid),
        Op::Softplus(a) => value(a).map(special::softplus),
        Op::Gammaln(a) => value(a).map(special::gammaln),
        Op::Xlogy(a, b) => value(a)
            .binary(value(b), special::xlogy)
            .expect("shapes broadcast"),
        Op::Sum(a) => Tensor::scalar(value(a).sum()),
        Op::Gather(a, map) => {
            let parent = value(a);
            let data: Vec<f64> = map.map.iter().map(|&i| parent.data()[i]).collect();
            Tensor::from_vec(map.out_shape.clone(), data)
        }
        Op::Scatter { len, parts } => {
            let mut data = vec![0.0; *len];
            for (var, positions) in parts {
                let src = value(var);
                assert_eq!(
                    src.len(),
                    positions.len(),
                    "scatter segment length mismatch"
                );
                for (v, &pos) in src.data().iter().zip(positions.iter()) {
                    data[pos] = *v;
                }
            }
            Tensor::from_vec(vec![*len], data)
        }
        Op::Where {
            cond,
            then_v,
            else_v,
        } => {
            let shape = Tensor::broadcast_shapes(value(cond).shape(), value(then_v).shape())
                .and_then(|s| Tensor::broadcast_shapes(&s, value(else_v).shape()))
                .expect("shapes broadcast");
            let cond_b = value(cond).broadcast_to(&shape).expect("cond broadcasts");
            let then_b = value(then_v).broadcast_to(&shape).expect("then broadcasts");
            let else_b = value(else_v).broadcast_to(&shape).expect("else broadcasts");
            let data: Vec<f64> = cond_b
                .data()
                .iter()
                .zip(then_b.data().iter().zip(else_b.data()))
                .map(|(&c, (&t, &e))| if c != 0.0 { t } else { e })
                .collect();
            Tensor::from_vec(shape, data)
        }
        Op::OrderedInverse(a) => {
            let y = value(a);
            let mut data = Vec::with_capacity(y.len());
            let mut acc = y.data()[0];
            data.push(acc);
            for &yi in &y.data()[1..] {
                acc += yi.exp();
                data.push(acc);
            }
            Tensor::from_vec(vec![y.len()], data)
        }
        Op::NormalLogProb {
            value: x,
            loc,
            scale,
        } => {
            let shape = Tensor::broadcast_shapes(value(x).shape(), value(loc).shape())
                .and_then(|shape| Tensor::broadcast_shapes(&shape, value(scale).shape()))
                .expect("Normal log_prob shapes broadcast");
            let value_b = value(x).broadcast_to(&shape).expect("value broadcasts");
            let loc_b = value(loc).broadcast_to(&shape).expect("loc broadcasts");
            let scale_b = value(scale).broadcast_to(&shape).expect("scale broadcasts");
            let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
            let data = value_b
                .data()
                .iter()
                .zip(loc_b.data())
                .zip(scale_b.data())
                .map(|((&x, &m), &s)| {
                    let delta = x - m;
                    let standardized = delta / s;
                    let sq = standardized * standardized;
                    let term = -0.5 * sq;
                    let term = term - s.ln();
                    term - half_log_2pi
                })
                .collect();
            Tensor::from_vec(shape, data)
        }
        Op::MultivariateNormalLogProb {
            value: x,
            mean,
            scale_tril,
        } => {
            let l_t = value(scale_tril);
            let n = l_t.shape()[0];
            let value_b = value(x)
                .broadcast_to(&[n])
                .expect("value broadcasts to event shape");
            let mean_b = value(mean)
                .broadcast_to(&[n])
                .expect("mean broadcasts to event shape");
            let delta: Vec<f64> = value_b
                .data()
                .iter()
                .zip(mean_b.data())
                .map(|(&x, &m)| x - m)
                .collect();
            let solved = linalg::solve_lower(n, l_t.data(), &delta);
            let quadratic: f64 = solved.iter().map(|z| z * z).sum();
            let log_det: f64 = (0..n).map(|i| l_t.data()[i * n + i].ln()).sum();
            let term = -0.5 * quadratic;
            let term = term - log_det;
            let logp = term - 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();
            Tensor::scalar(logp)
        }
        Op::SolveLower(l, b) => {
            let l_t = value(l);
            let b_t = value(b);
            let n = l_t.shape()[0];
            let x = linalg::solve_lower(n, l_t.data(), b_t.data());
            Tensor::from_vec(vec![n], x)
        }
        Op::ConcatLast(parts) => {
            let first = value(&parts[0]);
            let lead = first.shape()[..first.rank() - 1].to_vec();
            let rows: usize = lead.iter().product();
            let mut last_total = 0usize;
            for part in parts {
                let t = value(part);
                last_total += t.shape()[t.rank() - 1];
            }
            let mut data = Vec::with_capacity(rows * last_total);
            for row in 0..rows {
                for part in parts {
                    let t = value(part);
                    let w = t.shape()[t.rank() - 1];
                    data.extend_from_slice(&t.data()[row * w..(row + 1) * w]);
                }
            }
            let mut shape = lead;
            shape.push(last_total);
            Tensor::from_vec(shape, data)
        }
        Op::Broadcast(a, shape) => value(a).broadcast_to(shape).expect("shape broadcasts"),
        Op::Reshape(a, shape) => value(a)
            .reshape(shape.clone())
            .expect("reshape size matches"),
        Op::Cmp(kind, a, b) => value(a)
            .binary(value(b), |x, y| {
                let holds = match kind {
                    CmpKind::Ge => x >= y,
                    CmpKind::Gt => x > y,
                    CmpKind::Le => x <= y,
                    CmpKind::Lt => x < y,
                };
                if holds {
                    1.0
                } else {
                    0.0
                }
            })
            .expect("shapes broadcast"),
        Op::And(a, b) => value(a)
            .binary(
                value(b),
                |x, y| {
                    if x != 0.0 && y != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                },
            )
            .expect("shapes broadcast"),
        Op::IsInteger(a) => value(a).map(|x| if x == x.floor() { 1.0 } else { 0.0 }),
        Op::IsStrictlyIncreasing(a) => {
            let data = value(a).data();
            let increasing = data.windows(2).all(|w| w[1] > w[0]);
            Tensor::scalar(if increasing { 1.0 } else { 0.0 })
        }
        Op::TakeAlongLast { base, index } => {
            let base_t = value(base);
            let idx = value(index);
            let k = *base_t.shape().last().expect("base has rank >= 1");
            let data: Vec<f64> = idx
                .data()
                .iter()
                .enumerate()
                .map(|(row, &raw)| {
                    let clipped = raw.clamp(0.0, (k - 1) as f64);
                    base_t.data()[row * k + clipped as usize]
                })
                .collect();
            Tensor::from_vec(idx.shape().to_vec(), data)
        }
    }
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape { nodes: Vec::new() }
    }

    pub fn value(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].value
    }

    pub fn requires_grad(&self, v: Var) -> bool {
        self.nodes[v.0].requires_grad
    }

    /// Whether any parent (through any op, including grad-free masks)
    /// depends on an input leaf.
    fn op_dynamic(&self, op: &Op) -> bool {
        let dynamic = |v: &Var| self.nodes[v.0].dynamic;
        match op {
            Op::Leaf => false,
            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
            | Op::Div(a, b)
            | Op::Xlogy(a, b)
            | Op::SolveLower(a, b)
            | Op::Cmp(_, a, b)
            | Op::And(a, b) => dynamic(a) || dynamic(b),
            Op::Neg(a)
            | Op::Exp(a)
            | Op::Ln(a)
            | Op::Ln1p(a)
            | Op::Sigmoid(a)
            | Op::Softplus(a)
            | Op::Gammaln(a)
            | Op::Sum(a)
            | Op::Gather(a, _)
            | Op::OrderedInverse(a)
            | Op::Broadcast(a, _)
            | Op::Reshape(a, _)
            | Op::IsInteger(a)
            | Op::IsStrictlyIncreasing(a) => dynamic(a),
            Op::Scatter { parts, .. } => parts.iter().any(|(v, _)| dynamic(v)),
            Op::Where {
                cond,
                then_v,
                else_v,
            } => dynamic(cond) || dynamic(then_v) || dynamic(else_v),
            Op::NormalLogProb { value, loc, scale } => {
                dynamic(value) || dynamic(loc) || dynamic(scale)
            }
            Op::MultivariateNormalLogProb {
                value,
                mean,
                scale_tril,
            } => dynamic(value) || dynamic(mean) || dynamic(scale_tril),
            Op::ConcatLast(parts) => parts.iter().any(dynamic),
            Op::TakeAlongLast { base, index } => dynamic(base) || dynamic(index),
        }
    }

    /// Evaluate `op` from its parents' current values and push the node.
    fn push_op(&mut self, op: Op, requires_grad: bool) -> Var {
        let value = eval_op(&self.nodes, &op);
        let dynamic = self.op_dynamic(&op);
        self.nodes.push(Node {
            value,
            op,
            requires_grad,
            dynamic,
        });
        Var(self.nodes.len() - 1)
    }

    /// A constant leaf (no gradient, never recomputed).
    pub fn constant(&mut self, value: Tensor) -> Var {
        self.nodes.push(Node {
            value,
            op: Op::Leaf,
            requires_grad: false,
            dynamic: false,
        });
        Var(self.nodes.len() - 1)
    }

    /// An input leaf that participates in gradients and can be updated
    /// between replays via [`Tape::set_leaf`].
    pub fn input(&mut self, value: Tensor) -> Var {
        self.nodes.push(Node {
            value,
            op: Op::Leaf,
            requires_grad: true,
            dynamic: true,
        });
        Var(self.nodes.len() - 1)
    }

    /// Overwrite an input leaf's data in place (same shape).
    pub fn set_leaf(&mut self, v: Var, data: &[f64]) {
        let node = &mut self.nodes[v.0];
        debug_assert!(matches!(node.op, Op::Leaf), "set_leaf targets leaves");
        node.value.data_mut().copy_from_slice(data);
    }

    /// Re-run the forward pass in evaluation order, recomputing every node
    /// that depends on an input leaf. Uses the same `eval_op` code path as
    /// graph construction, so results are bit-identical to a fresh build.
    pub fn replay(&mut self) {
        for i in 0..self.nodes.len() {
            let (prev, rest) = self.nodes.split_at_mut(i);
            let node = &mut rest[0];
            if matches!(node.op, Op::Leaf) || !node.dynamic {
                continue;
            }
            node.value = eval_op(prev, &node.op);
        }
    }

    fn binary_grad(&self, a: Var, b: Var) -> bool {
        self.requires_grad(a) || self.requires_grad(b)
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let grad = self.binary_grad(a, b);
        self.push_op(Op::Add(a, b), grad)
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let grad = self.binary_grad(a, b);
        self.push_op(Op::Sub(a, b), grad)
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let grad = self.binary_grad(a, b);
        self.push_op(Op::Mul(a, b), grad)
    }

    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let grad = self.binary_grad(a, b);
        self.push_op(Op::Div(a, b), grad)
    }

    pub fn neg(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Neg(a), grad)
    }

    pub fn exp(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Exp(a), grad)
    }

    pub fn ln(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Ln(a), grad)
    }

    pub fn ln_1p(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Ln1p(a), grad)
    }

    pub fn sigmoid(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Sigmoid(a), grad)
    }

    pub fn softplus(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Softplus(a), grad)
    }

    pub fn gammaln(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Gammaln(a), grad)
    }

    pub fn xlogy(&mut self, a: Var, b: Var) -> Var {
        let grad = self.binary_grad(a, b);
        self.push_op(Op::Xlogy(a, b), grad)
    }

    pub fn sum(&mut self, a: Var) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Sum(a), grad)
    }

    pub fn gather(&mut self, a: Var, map: GatherMap) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Gather(a, map), grad)
    }

    /// Assemble a rank-1 vector of length `len` from scattered segments.
    /// Positions must be in-bounds; later writes win on overlap (JAX
    /// `.at[].set` chaining), though the IR validates disjointness upstream.
    pub fn scatter(&mut self, len: usize, parts: Vec<(Var, Vec<usize>)>) -> Var {
        let grad = parts.iter().any(|(var, _)| self.requires_grad(*var));
        self.push_op(Op::Scatter { len, parts }, grad)
    }

    /// Elementwise select with broadcasting; gradient flows through the
    /// selected branch only (select semantics, not masked multiply, so an
    /// infinite adjoint cannot poison the unselected branch). The condition
    /// is a tape var — typically a grad-free mask — so it is recomputed when
    /// the tape is replayed at a new point.
    pub fn where_select(&mut self, cond: Var, then_v: Var, else_v: Var) -> Var {
        let grad = self.binary_grad(then_v, else_v);
        self.push_op(
            Op::Where {
                cond,
                then_v,
                else_v,
            },
            grad,
        )
    }

    /// Grad-free elementwise `a >= b` mask (1.0/0.0) with broadcasting.
    pub fn ge(&mut self, a: Var, b: Var) -> Result<Var, Error> {
        self.cmp(CmpKind::Ge, a, b)
    }

    /// Grad-free elementwise `a > b` mask (1.0/0.0) with broadcasting.
    pub fn gt(&mut self, a: Var, b: Var) -> Result<Var, Error> {
        self.cmp(CmpKind::Gt, a, b)
    }

    /// Grad-free elementwise `a <= b` mask (1.0/0.0) with broadcasting.
    pub fn le(&mut self, a: Var, b: Var) -> Result<Var, Error> {
        self.cmp(CmpKind::Le, a, b)
    }

    /// Grad-free elementwise `a < b` mask (1.0/0.0) with broadcasting.
    pub fn lt(&mut self, a: Var, b: Var) -> Result<Var, Error> {
        self.cmp(CmpKind::Lt, a, b)
    }

    fn cmp(&mut self, kind: CmpKind, a: Var, b: Var) -> Result<Var, Error> {
        Tensor::broadcast_shapes(self.value(a).shape(), self.value(b).shape())?;
        Ok(self.push_op(Op::Cmp(kind, a, b), false))
    }

    /// Grad-free logical AND of two 0.0/1.0 masks with broadcasting.
    pub fn and(&mut self, a: Var, b: Var) -> Result<Var, Error> {
        Tensor::broadcast_shapes(self.value(a).shape(), self.value(b).shape())?;
        Ok(self.push_op(Op::And(a, b), false))
    }

    /// Grad-free elementwise "is an integer" mask (1.0/0.0).
    pub fn is_integer(&mut self, a: Var) -> Var {
        self.push_op(Op::IsInteger(a), false)
    }

    /// Grad-free scalar mask: 1.0 iff the rank-1 input strictly increases.
    pub fn is_strictly_increasing(&mut self, a: Var) -> Var {
        assert_eq!(self.value(a).rank(), 1, "expects a rank-1 input");
        self.push_op(Op::IsStrictlyIncreasing(a), false)
    }

    /// `out[row] = base[row, clamp(index[row], 0, k-1)]` along the last axis
    /// of `base`. Indices are re-read per evaluation; gradient flows through
    /// `base` only.
    pub fn take_along_last(&mut self, base: Var, index: Var) -> Var {
        let base_shape = self.value(base).shape();
        assert!(!base_shape.is_empty(), "base must have rank >= 1");
        assert_eq!(
            &base_shape[..base_shape.len() - 1],
            self.value(index).shape(),
            "index shape must equal base leading dims"
        );
        let grad = self.requires_grad(base);
        self.push_op(Op::TakeAlongLast { base, index }, grad)
    }

    pub fn ordered_inverse(&mut self, a: Var) -> Var {
        assert_eq!(
            self.value(a).rank(),
            1,
            "Ordered constraint requires vector values"
        );
        let grad = self.requires_grad(a);
        self.push_op(Op::OrderedInverse(a), grad)
    }

    /// Elementwise Normal log probability, materialized at the broadcasted shape.
    pub fn normal_log_prob(&mut self, value: Var, loc: Var, scale: Var) -> Var {
        let grad =
            self.requires_grad(value) || self.requires_grad(loc) || self.requires_grad(scale);
        self.push_op(Op::NormalLogProb { value, loc, scale }, grad)
    }

    /// MultivariateNormal log probability for a single rank-1 event.
    pub fn multivariate_normal_log_prob(&mut self, value: Var, mean: Var, scale_tril: Var) -> Var {
        let l_t = self.value(scale_tril);
        assert_eq!(l_t.rank(), 2, "scale_tril must be rank-2");
        let n = l_t.shape()[0];
        assert_eq!(l_t.shape(), &[n, n], "scale_tril must be square");
        let grad =
            self.requires_grad(value) || self.requires_grad(mean) || self.requires_grad(scale_tril);
        self.push_op(
            Op::MultivariateNormalLogProb {
                value,
                mean,
                scale_tril,
            },
            grad,
        )
    }

    /// Solve `L x = b`, L rank-2 lower-triangular, b rank-1.
    pub fn solve_lower(&mut self, l: Var, b: Var) -> Var {
        let l_t = self.value(l);
        assert_eq!(l_t.rank(), 2);
        let n = l_t.shape()[0];
        assert_eq!(l_t.shape(), &[n, n]);
        assert_eq!(self.value(b).shape(), &[n]);
        let grad = self.binary_grad(l, b);
        self.push_op(Op::SolveLower(l, b), grad)
    }

    /// Concatenate along the last axis; all parts share leading dims.
    pub fn concat_last(&mut self, parts: Vec<Var>) -> Var {
        assert!(!parts.is_empty());
        let lead = self.value(parts[0]).shape()[..self.value(parts[0]).rank() - 1].to_vec();
        let mut grad = false;
        for &part in &parts {
            let t = self.value(part);
            assert!(t.rank() >= 1, "concat_last expects rank >= 1");
            assert_eq!(
                &t.shape()[..t.rank() - 1],
                lead.as_slice(),
                "leading dims differ"
            );
            grad |= self.requires_grad(part);
        }
        self.push_op(Op::ConcatLast(parts), grad)
    }

    /// Materialize a broadcast of `a` to `shape`.
    pub fn broadcast(&mut self, a: Var, shape: &[usize]) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Broadcast(a, shape.to_vec()), grad)
    }

    pub fn reshape(&mut self, a: Var, shape: Vec<usize>) -> Var {
        let grad = self.requires_grad(a);
        self.push_op(Op::Reshape(a, shape), grad)
    }

    /// Reverse pass from a scalar root; returns per-node adjoints for the
    /// requested leaves.
    pub fn backward(&self, root: Var, leaves: &[Var]) -> Vec<Tensor> {
        let mut adjoints = Vec::new();
        self.backward_into(root, leaves, &mut adjoints)
    }

    /// [`Tape::backward`] with a caller-owned adjoint slot buffer, so
    /// repeated evaluations reuse its allocation instead of allocating one
    /// slot vector per gradient.
    pub fn backward_into(
        &self,
        root: Var,
        leaves: &[Var],
        adjoints: &mut Vec<Option<Tensor>>,
    ) -> Vec<Tensor> {
        assert_eq!(self.value(root).len(), 1, "backward needs a scalar root");
        adjoints.clear();
        adjoints.resize(self.nodes.len(), None);
        adjoints[root.0] = Some(Tensor::scalar(1.0));

        for id in (0..=root.0).rev() {
            if !self.nodes[id].requires_grad {
                continue;
            }
            let Some(adj) = adjoints[id].take() else {
                continue;
            };
            self.propagate(id, &adj, adjoints);
            adjoints[id] = Some(adj);
        }

        leaves
            .iter()
            .map(|leaf| {
                adjoints[leaf.0]
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros(self.value(*leaf).shape()))
            })
            .collect()
    }

    fn accumulate(&self, adjoints: &mut [Option<Tensor>], var: Var, contribution: Tensor) {
        if !self.nodes[var.0].requires_grad {
            return;
        }
        // Reduce broadcasting before accumulating.
        let reduced = contribution.reduce_to_shape(self.value(var).shape());
        match &mut adjoints[var.0] {
            Some(existing) => {
                let updated = existing.binary(&reduced, |x, y| x + y).expect("same shape");
                *existing = updated;
            }
            slot @ None => *slot = Some(reduced),
        }
    }

    fn propagate(&self, id: usize, adj: &Tensor, adjoints: &mut [Option<Tensor>]) {
        match &self.nodes[id].op {
            Op::Leaf => {}
            // Grad-free masks never require gradients, so propagate is never
            // called on them.
            Op::Cmp(..) | Op::And(..) | Op::IsInteger(..) | Op::IsStrictlyIncreasing(..) => {}
            Op::Add(a, b) => {
                self.accumulate(adjoints, *a, adj.clone());
                self.accumulate(adjoints, *b, adj.clone());
            }
            Op::Sub(a, b) => {
                self.accumulate(adjoints, *a, adj.clone());
                self.accumulate(adjoints, *b, adj.map(|x| -x));
            }
            Op::Mul(a, b) => {
                let da = adj.binary(self.value(*b), |g, y| g * y).expect("broadcast");
                let db = adj.binary(self.value(*a), |g, x| g * x).expect("broadcast");
                self.accumulate(adjoints, *a, da);
                self.accumulate(adjoints, *b, db);
            }
            Op::Div(a, b) => {
                let da = adj.binary(self.value(*b), |g, y| g / y).expect("broadcast");
                self.accumulate(adjoints, *a, da);
                if self.requires_grad(*b) {
                    // d/db (a/b) = -a / b^2
                    let out = &self.nodes[id].value; // a/b
                    let db = adj
                        .binary(out, |g, q| g * q)
                        .expect("broadcast")
                        .binary(self.value(*b), |gq, y| -gq / y)
                        .expect("broadcast");
                    self.accumulate(adjoints, *b, db);
                }
            }
            Op::Neg(a) => self.accumulate(adjoints, *a, adj.map(|x| -x)),
            Op::Exp(a) => {
                let out = &self.nodes[id].value;
                let da = adj.binary(out, |g, e| g * e).expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Ln(a) => {
                let da = adj
                    .binary(self.value(*a), |g, x| g / x)
                    .expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Ln1p(a) => {
                let da = adj
                    .binary(self.value(*a), |g, x| g / (1.0 + x))
                    .expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Sigmoid(a) => {
                let out = &self.nodes[id].value;
                let da = adj
                    .binary(out, |g, s| g * s * (1.0 - s))
                    .expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Softplus(a) => {
                let da = adj
                    .binary(self.value(*a), |g, x| g * special::sigmoid(x))
                    .expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Gammaln(a) => {
                let da = adj
                    .binary(self.value(*a), |g, x| g * special::digamma(x))
                    .expect("same shape");
                self.accumulate(adjoints, *a, da);
            }
            Op::Xlogy(a, b) => {
                if self.requires_grad(*a) {
                    let da = adj
                        .binary(self.value(*b), |g, y| g * y.ln())
                        .expect("broadcast");
                    self.accumulate(adjoints, *a, da);
                }
                if self.requires_grad(*b) {
                    let ratio = self
                        .value(*a)
                        .binary(self.value(*b), |x, y| if x == 0.0 { 0.0 } else { x / y })
                        .expect("broadcast");
                    let db = adj.binary(&ratio, |g, r| g * r).expect("broadcast");
                    self.accumulate(adjoints, *b, db);
                }
            }
            Op::Sum(a) => {
                let g = adj.data()[0];
                let shape = self.value(*a).shape().to_vec();
                let data = vec![g; self.value(*a).len()];
                self.accumulate(adjoints, *a, Tensor::from_vec(shape, data));
            }
            Op::Gather(a, map) => {
                let mut grad = Tensor::zeros(self.value(*a).shape());
                for (g, &src) in adj.data().iter().zip(map.map.iter()) {
                    grad.data_mut()[src] += g;
                }
                self.accumulate(adjoints, *a, grad);
            }
            Op::Scatter { parts, .. } => {
                for (var, positions) in parts {
                    if !self.requires_grad(*var) {
                        continue;
                    }
                    let data: Vec<f64> = positions.iter().map(|&p| adj.data()[p]).collect();
                    let shape = self.value(*var).shape().to_vec();
                    self.accumulate(adjoints, *var, Tensor::from_vec(shape, data));
                }
            }
            Op::Where {
                cond,
                then_v,
                else_v,
            } => {
                let cond_b = self
                    .value(*cond)
                    .broadcast_to(self.nodes[id].value.shape())
                    .expect("cond broadcasts");
                if self.requires_grad(*then_v) {
                    let data: Vec<f64> = cond_b
                        .data()
                        .iter()
                        .zip(adj.data())
                        .map(|(&c, &g)| if c != 0.0 { g } else { 0.0 })
                        .collect();
                    self.accumulate(
                        adjoints,
                        *then_v,
                        Tensor::from_vec(adj.shape().to_vec(), data),
                    );
                }
                if self.requires_grad(*else_v) {
                    let data: Vec<f64> = cond_b
                        .data()
                        .iter()
                        .zip(adj.data())
                        .map(|(&c, &g)| if c != 0.0 { 0.0 } else { g })
                        .collect();
                    self.accumulate(
                        adjoints,
                        *else_v,
                        Tensor::from_vec(adj.shape().to_vec(), data),
                    );
                }
            }
            Op::OrderedInverse(a) => {
                // x[i] = y[0] + sum_{1<=k<=i} exp(y[k])
                // dy[0] = sum_i adj[i]; dy[k] = exp(y[k]) * sum_{i>=k} adj[i].
                let y = self.value(*a);
                let n = y.len();
                let mut suffix = vec![0.0; n];
                let mut acc = 0.0;
                for i in (0..n).rev() {
                    acc += adj.data()[i];
                    suffix[i] = acc;
                }
                let mut grad = vec![0.0; n];
                grad[0] = suffix[0];
                for k in 1..n {
                    grad[k] = y.data()[k].exp() * suffix[k];
                }
                self.accumulate(adjoints, *a, Tensor::from_vec(vec![n], grad));
            }
            Op::NormalLogProb { value, loc, scale } => {
                let shape = self.nodes[id].value.shape().to_vec();
                let value_b = self
                    .value(*value)
                    .broadcast_to(&shape)
                    .expect("value broadcasts");
                let loc_b = self
                    .value(*loc)
                    .broadcast_to(&shape)
                    .expect("loc broadcasts");
                let scale_b = self
                    .value(*scale)
                    .broadcast_to(&shape)
                    .expect("scale broadcasts");
                if self.requires_grad(*value) {
                    let data = adj
                        .data()
                        .iter()
                        .zip(value_b.data().iter().zip(loc_b.data()).zip(scale_b.data()))
                        .map(|(&g, ((&x, &m), &s))| {
                            let delta = x - m;
                            -g * delta / (s * s)
                        })
                        .collect();
                    self.accumulate(adjoints, *value, Tensor::from_vec(shape.clone(), data));
                }
                if self.requires_grad(*loc) {
                    let data = adj
                        .data()
                        .iter()
                        .zip(value_b.data().iter().zip(loc_b.data()).zip(scale_b.data()))
                        .map(|(&g, ((&x, &m), &s))| {
                            let delta = x - m;
                            g * delta / (s * s)
                        })
                        .collect();
                    self.accumulate(adjoints, *loc, Tensor::from_vec(shape.clone(), data));
                }
                if self.requires_grad(*scale) {
                    let data = adj
                        .data()
                        .iter()
                        .zip(value_b.data().iter().zip(loc_b.data()).zip(scale_b.data()))
                        .map(|(&g, ((&x, &m), &s))| {
                            let delta = x - m;
                            g * (delta * delta / (s * s * s) - 1.0 / s)
                        })
                        .collect();
                    self.accumulate(adjoints, *scale, Tensor::from_vec(shape, data));
                }
            }
            Op::MultivariateNormalLogProb {
                value,
                mean,
                scale_tril,
            } => {
                let g = adj.data()[0];
                let l_t = self.value(*scale_tril);
                let n = l_t.shape()[0];
                let value_b = self
                    .value(*value)
                    .broadcast_to(&[n])
                    .expect("value broadcasts to event shape");
                let mean_b = self
                    .value(*mean)
                    .broadcast_to(&[n])
                    .expect("mean broadcasts to event shape");
                let delta: Vec<f64> = value_b
                    .data()
                    .iter()
                    .zip(mean_b.data())
                    .map(|(&x, &m)| x - m)
                    .collect();
                let z = linalg::solve_lower(n, l_t.data(), &delta);
                let alpha = linalg::solve_lower_transpose(n, l_t.data(), &z);
                if self.requires_grad(*value) {
                    self.accumulate(
                        adjoints,
                        *value,
                        Tensor::from_vec(vec![n], alpha.iter().map(|a| -g * a).collect()),
                    );
                }
                if self.requires_grad(*mean) {
                    self.accumulate(
                        adjoints,
                        *mean,
                        Tensor::from_vec(vec![n], alpha.iter().map(|a| g * a).collect()),
                    );
                }
                if self.requires_grad(*scale_tril) {
                    let mut dl = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..=i {
                            dl[i * n + j] = g * alpha[i] * z[j];
                        }
                        dl[i * n + i] -= g / l_t.data()[i * n + i];
                    }
                    self.accumulate(adjoints, *scale_tril, Tensor::from_vec(vec![n, n], dl));
                }
            }
            Op::SolveLower(l, b) => {
                // x = L^{-1} b. db = L^{-T} adj; dL = -db x^T (lower part).
                let l_t = self.value(*l);
                let n = l_t.shape()[0];
                let x = self.nodes[id].value.data();
                let db = linalg::solve_lower_transpose(n, l_t.data(), adj.data());
                if self.requires_grad(*b) {
                    self.accumulate(adjoints, *b, Tensor::from_vec(vec![n], db.clone()));
                }
                if self.requires_grad(*l) {
                    let mut dl = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..=i {
                            dl[i * n + j] = -db[i] * x[j];
                        }
                    }
                    self.accumulate(adjoints, *l, Tensor::from_vec(vec![n, n], dl));
                }
            }
            Op::ConcatLast(parts) => {
                let out = &self.nodes[id].value;
                let out_w = out.shape()[out.rank() - 1];
                let rows = out.len() / out_w;
                let mut offset = 0usize;
                for var in parts {
                    let t = self.value(*var);
                    let w = t.shape()[t.rank() - 1];
                    if self.requires_grad(*var) {
                        let mut data = Vec::with_capacity(t.len());
                        for row in 0..rows {
                            let start = row * out_w + offset;
                            data.extend_from_slice(&adj.data()[start..start + w]);
                        }
                        self.accumulate(adjoints, *var, Tensor::from_vec(t.shape().to_vec(), data));
                    }
                    offset += w;
                }
            }
            Op::Broadcast(a, _) => {
                self.accumulate(adjoints, *a, adj.clone());
            }
            Op::Reshape(a, _) => {
                let shape = self.value(*a).shape().to_vec();
                let grad = Tensor::from_vec(shape, adj.data().to_vec());
                self.accumulate(adjoints, *a, grad);
            }
            Op::TakeAlongLast { base, index } => {
                if self.requires_grad(*base) {
                    let base_t = self.value(*base);
                    let idx = self.value(*index);
                    let k = *base_t.shape().last().expect("base has rank >= 1");
                    let mut grad = Tensor::zeros(base_t.shape());
                    for (row, (&g, &raw)) in adj.data().iter().zip(idx.data()).enumerate() {
                        let clipped = raw.clamp(0.0, (k - 1) as f64);
                        grad.data_mut()[row * k + clipped as usize] += g;
                    }
                    self.accumulate(adjoints, *base, grad);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{gather_map, IndexAtom};

    /// Central finite-difference check of d(scalar fn)/d(inputs).
    fn grad_check(build: impl Fn(&mut Tape, &[Var]) -> Var, inputs: &[Tensor], tol: f64) {
        let mut tape = Tape::new();
        let vars: Vec<Var> = inputs.iter().map(|t| tape.input(t.clone())).collect();
        let root = build(&mut tape, &vars);
        let grads = tape.backward(root, &vars);

        let eps = 1e-6;
        for (which, input) in inputs.iter().enumerate() {
            for elem in 0..input.len() {
                let mut plus = inputs.to_vec();
                plus[which].data_mut()[elem] += eps;
                let mut minus = inputs.to_vec();
                minus[which].data_mut()[elem] -= eps;

                let eval = |ins: &[Tensor]| -> f64 {
                    let mut t = Tape::new();
                    let vs: Vec<Var> = ins.iter().map(|x| t.input(x.clone())).collect();
                    let r = build(&mut t, &vs);
                    t.value(r).data()[0]
                };
                let numeric = (eval(&plus) - eval(&minus)) / (2.0 * eps);
                let analytic = grads[which].data()[elem];
                assert!(
                    (numeric - analytic).abs() <= tol * (1.0 + numeric.abs()),
                    "grad mismatch input {which} elem {elem}: analytic {analytic}, numeric {numeric}"
                );
            }
        }
    }

    #[test]
    fn arithmetic_gradients() {
        grad_check(
            |t, v| {
                let p = t.mul(v[0], v[1]);
                let q = t.div(v[0], v[1]);
                let s = t.sub(p, q);
                let n = t.neg(s);
                let a = t.add(n, v[0]);
                t.sum(a)
            },
            &[
                Tensor::from_vec(vec![3], vec![0.5, -1.2, 2.0]),
                Tensor::from_vec(vec![3], vec![1.5, 0.7, -0.4]),
            ],
            1e-7,
        );
    }

    #[test]
    fn broadcast_gradients_reduce() {
        // scalar * vector: scalar grad must sum over the vector.
        grad_check(
            |t, v| {
                let p = t.mul(v[0], v[1]);
                t.sum(p)
            },
            &[
                Tensor::scalar(1.3),
                Tensor::from_vec(vec![4], vec![1.0, -2.0, 3.0, 0.5]),
            ],
            1e-7,
        );
    }

    #[test]
    fn unary_gradients() {
        grad_check(
            |t, v| {
                let e = t.exp(v[0]);
                let l = t.ln(e);
                let lp = t.ln_1p(l);
                let sg = t.sigmoid(lp);
                let sp = t.softplus(sg);
                t.sum(sp)
            },
            &[Tensor::from_vec(vec![3], vec![0.3, -0.8, 1.7])],
            1e-6,
        );
    }

    #[test]
    fn gammaln_gradient_is_digamma() {
        grad_check(
            |t, v| {
                let g = t.gammaln(v[0]);
                t.sum(g)
            },
            &[Tensor::from_vec(vec![3], vec![0.7, 2.5, 11.0])],
            1e-5,
        );
    }

    #[test]
    fn xlogy_gradients() {
        grad_check(
            |t, v| {
                let x = t.xlogy(v[0], v[1]);
                t.sum(x)
            },
            &[
                Tensor::from_vec(vec![2], vec![3.0, 0.5]),
                Tensor::from_vec(vec![2], vec![0.25, 1.5]),
            ],
            1e-6,
        );
    }

    #[test]
    fn normal_log_prob_scalar_value_and_gradients_match_formula() {
        let mut tape = Tape::new();
        let value = tape.input(Tensor::scalar(1.25));
        let loc = tape.input(Tensor::scalar(0.5));
        let scale = tape.input(Tensor::scalar(2.0));
        let lp = tape.normal_log_prob(value, loc, scale);
        let root = tape.sum(lp);
        let grads = tape.backward(root, &[value, loc, scale]);

        let delta = 1.25 - 0.5;
        let expected = -0.5 * (delta / 2.0) * (delta / 2.0)
            - 2.0f64.ln()
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((tape.value(root).data()[0] - expected).abs() < 1e-12);
        assert!((grads[0].data()[0] - (-delta / 4.0)).abs() < 1e-12);
        assert!((grads[1].data()[0] - (delta / 4.0)).abs() < 1e-12);
        assert!((grads[2].data()[0] - (delta * delta / 8.0 - 0.5)).abs() < 1e-12);
    }

    #[test]
    fn normal_log_prob_broadcast_gradients_reduce() {
        grad_check(
            |t, v| {
                let lp = t.normal_log_prob(v[0], v[1], v[2]);
                t.sum(lp)
            },
            &[
                Tensor::from_vec(vec![3], vec![0.5, -1.0, 2.0]),
                Tensor::scalar(0.25),
                Tensor::from_vec(vec![3], vec![1.5, 0.7, 2.2]),
            ],
            1e-6,
        );
    }

    #[test]
    fn gather_and_scatter_gradients() {
        let map = gather_map(
            &[3],
            &[IndexAtom::Array {
                shape: vec![4],
                values: vec![2, 0, 1, 2],
            }],
        )
        .unwrap();
        grad_check(
            move |t, v| {
                let g = t.gather(v[0], map.clone());
                let s = t.scatter(5, vec![(g, vec![4, 0, 2, 1])]);
                let sq = t.mul(s, s);
                t.sum(sq)
            },
            &[Tensor::from_vec(vec![3], vec![0.5, -1.0, 2.0])],
            1e-6,
        );
    }

    #[test]
    fn where_routes_gradient_to_selected_branch() {
        let cond = Tensor::from_vec(vec![3], vec![1.0, 0.0, 1.0]);
        grad_check(
            move |t, v| {
                let c = t.constant(cond.clone());
                let w = t.where_select(c, v[0], v[1]);
                t.sum(w)
            },
            &[
                Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
                Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]),
            ],
            1e-9,
        );
        // An infinite unselected branch must not poison the gradient.
        let mut tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2], vec![1.0, 2.0]));
        let inf = tape.constant(Tensor::from_vec(vec![2], vec![f64::NEG_INFINITY; 2]));
        let cond = tape.constant(Tensor::from_vec(vec![2], vec![1.0, 1.0]));
        let w = tape.where_select(cond, a, inf);
        let s = tape.sum(w);
        let grads = tape.backward(s, &[a]);
        assert_eq!(grads[0].data(), &[1.0, 1.0]);
    }

    #[test]
    fn ordered_inverse_gradient() {
        grad_check(
            |t, v| {
                let x = t.ordered_inverse(v[0]);
                let sq = t.mul(x, x);
                t.sum(sq)
            },
            &[Tensor::from_vec(vec![4], vec![-0.5, 0.3, -1.0, 0.8])],
            1e-6,
        );
    }

    #[test]
    fn multivariate_normal_log_prob_gradients_match_finite_difference() {
        grad_check(
            |t, v| t.multivariate_normal_log_prob(v[0], v[1], v[2]),
            &[
                Tensor::from_vec(vec![3], vec![0.7, -1.2, 0.5]),
                Tensor::from_vec(vec![3], vec![0.2, -0.4, 0.1]),
                Tensor::from_vec(
                    vec![3, 3],
                    vec![2.0, 0.0, 0.0, 0.6, 1.5, 0.0, -0.3, 0.4, 1.1],
                ),
            ],
            1e-5,
        );
    }

    #[test]
    fn multivariate_normal_log_prob_broadcast_mean_gradient_reduces() {
        grad_check(
            |t, v| t.multivariate_normal_log_prob(v[0], v[1], v[2]),
            &[
                Tensor::from_vec(vec![2], vec![0.7, -1.2]),
                Tensor::scalar(0.2),
                Tensor::from_vec(vec![2, 2], vec![1.3, 0.0, 0.4, 1.7]),
            ],
            1e-5,
        );
    }

    #[test]
    fn solve_lower_gradients() {
        let l = Tensor::from_vec(
            vec![3, 3],
            vec![2.0, 0.0, 0.0, 0.6, 1.5, 0.0, -0.3, 0.4, 1.1],
        );
        let b = Tensor::from_vec(vec![3], vec![0.7, -1.2, 0.5]);
        grad_check(
            |t, v| {
                let x = t.solve_lower(v[0], v[1]);
                let sq = t.mul(x, x);
                t.sum(sq)
            },
            &[l, b],
            1e-6,
        );
    }

    #[test]
    fn concat_and_reshape_gradients() {
        grad_check(
            |t, v| {
                let c = t.concat_last(vec![v[0], v[1]]);
                let r = t.reshape(c, vec![2, 2]);
                let sq = t.mul(r, r);
                t.sum(sq)
            },
            &[
                Tensor::from_vec(vec![2], vec![1.0, -2.0]),
                Tensor::from_vec(vec![2], vec![3.0, 0.5]),
            ],
            1e-6,
        );
    }

    #[test]
    fn constant_subtrees_get_no_gradient() {
        let mut tape = Tape::new();
        let c = tape.constant(Tensor::scalar(3.0));
        let x = tape.input(Tensor::scalar(2.0));
        let p = tape.mul(c, x);
        let s = tape.sum(p);
        let grads = tape.backward(s, &[x, c]);
        assert_eq!(grads[0].data(), &[3.0]);
        assert_eq!(grads[1].data(), &[0.0]); // constants report zero
    }

    #[test]
    fn take_along_last_selects_and_routes_gradient() {
        let mut tape = Tape::new();
        let base = tape.input(Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]));
        // Out-of-range indices clamp into [0, k-1], as in the JAX reference.
        let index = tape.constant(Tensor::from_vec(vec![2], vec![2.0, -1.0]));
        let taken = tape.take_along_last(base, index);
        assert_eq!(tape.value(taken).data(), &[3.0, 4.0]);
        let s = tape.sum(taken);
        let grads = tape.backward(s, &[base]);
        assert_eq!(grads[0].data(), &[0., 0., 1., 1., 0., 0.]);
    }

    #[test]
    fn replay_recomputes_masks_and_selected_branches() {
        // |x| via where(x >= 0, x, -x): replay at a negative point must flip
        // the mask, the selected branch, and the gradient sign.
        let mut tape = Tape::new();
        let x = tape.input(Tensor::scalar(1.5));
        let zero = tape.constant(Tensor::scalar(0.0));
        let cond = tape.ge(x, zero).unwrap();
        let neg = tape.neg(x);
        let w = tape.where_select(cond, x, neg);
        let s = tape.sum(w);
        assert_eq!(tape.value(s).data(), &[1.5]);
        assert_eq!(tape.backward(s, &[x])[0].data(), &[1.0]);

        tape.set_leaf(x, &[-2.0]);
        tape.replay();
        assert_eq!(tape.value(s).data(), &[2.0]);
        assert_eq!(tape.backward(s, &[x])[0].data(), &[-1.0]);

        // Replaying back at the original point restores the original values.
        tape.set_leaf(x, &[1.5]);
        tape.replay();
        assert_eq!(tape.value(s).data(), &[1.5]);
        assert_eq!(tape.backward(s, &[x])[0].data(), &[1.0]);
    }

    #[test]
    fn replay_matches_fresh_build_bitwise() {
        let build = |t: &mut Tape, v: Var| -> Var {
            let two = t.constant(Tensor::scalar(2.0));
            let scaled = t.mul(v, two);
            let e = t.exp(scaled);
            let sp = t.softplus(v);
            let g = t.gammaln(e);
            let zero = t.constant(Tensor::scalar(0.0));
            let mask = t.gt(v, zero).unwrap();
            let picked = t.where_select(mask, g, sp);
            t.sum(picked)
        };
        let q0 = Tensor::from_vec(vec![3], vec![0.5, -1.25, 2.0]);
        let q1 = Tensor::from_vec(vec![3], vec![-0.75, 0.3, 1.1]);

        let mut replayed = Tape::new();
        let x = replayed.input(q0.clone());
        let root = build(&mut replayed, x);
        replayed.set_leaf(x, q1.data());
        replayed.replay();

        let mut fresh = Tape::new();
        let y = fresh.input(q1);
        let fresh_root = build(&mut fresh, y);

        assert_eq!(replayed.value(root).data(), fresh.value(fresh_root).data());
        assert_eq!(
            replayed.backward(root, &[x])[0].data(),
            fresh.backward(fresh_root, &[y])[0].data()
        );
    }

    #[test]
    fn replay_skips_constant_subtrees() {
        let mut tape = Tape::new();
        let c = tape.constant(Tensor::scalar(3.0));
        let c2 = tape.exp(c);
        let x = tape.input(Tensor::scalar(1.0));
        let p = tape.mul(c2, x);
        let s = tape.sum(p);
        tape.set_leaf(x, &[2.0]);
        tape.replay();
        assert_eq!(tape.value(s).data(), &[2.0 * 3.0f64.exp()]);
        // The constant subtree keeps its value without recomputation
        // (dynamic=false); observable only through correctness here.
        assert_eq!(tape.value(c2).data(), &[3.0f64.exp()]);
    }
}
