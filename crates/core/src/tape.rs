//! Reverse-mode AD over the closed op set the IR compiles to.
//!
//! A `Tape` is built per evaluation: forward values are computed eagerly as
//! ops are pushed, and `backward` walks the tape once to accumulate adjoint
//! tensors. The op set is exactly what the evaluator and the distribution
//! log-densities need — this is not a general autodiff.

use crate::linalg;
use crate::special;
use crate::tensor::{GatherMap, Tensor};

/// Handle to a tape node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(usize);

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
        parts: Vec<(Var, Vec<usize>)>,
    },
    /// Elementwise select: cond ? then_v : else_v (all broadcast together).
    Where {
        cond: Tensor,
        then_v: Var,
        else_v: Var,
    },
    /// Ordered-constraint inverse: x[0]=y[0], x[i]=y[0]+sum_{k<=i} exp(y[k]).
    OrderedInverse(Var),
    /// Solve L x = b for lower-triangular L (rank-2) and rank-1 b.
    SolveLower(Var, Var),
    /// Concatenate along the last axis (equal leading dims).
    ConcatLast(Vec<Var>),
    /// Materialized broadcast to a wider shape.
    Broadcast(Var),
    Reshape(Var),
}

struct Node {
    value: Tensor,
    op: Op,
    requires_grad: bool,
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

    fn push(&mut self, value: Tensor, op: Op, requires_grad: bool) -> Var {
        self.nodes.push(Node {
            value,
            op,
            requires_grad,
        });
        Var(self.nodes.len() - 1)
    }

    /// A constant leaf (no gradient).
    pub fn constant(&mut self, value: Tensor) -> Var {
        self.push(value, Op::Leaf, false)
    }

    /// An input leaf that participates in gradients.
    pub fn input(&mut self, value: Tensor) -> Var {
        self.push(value, Op::Leaf, true)
    }

    fn binary_grad(&self, a: Var, b: Var) -> bool {
        self.requires_grad(a) || self.requires_grad(b)
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let value = self
            .value(a)
            .binary(self.value(b), |x, y| x + y)
            .expect("shapes broadcast");
        let grad = self.binary_grad(a, b);
        self.push(value, Op::Add(a, b), grad)
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let value = self
            .value(a)
            .binary(self.value(b), |x, y| x - y)
            .expect("shapes broadcast");
        let grad = self.binary_grad(a, b);
        self.push(value, Op::Sub(a, b), grad)
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let value = self
            .value(a)
            .binary(self.value(b), |x, y| x * y)
            .expect("shapes broadcast");
        let grad = self.binary_grad(a, b);
        self.push(value, Op::Mul(a, b), grad)
    }

    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let value = self
            .value(a)
            .binary(self.value(b), |x, y| x / y)
            .expect("shapes broadcast");
        let grad = self.binary_grad(a, b);
        self.push(value, Op::Div(a, b), grad)
    }

    pub fn neg(&mut self, a: Var) -> Var {
        let value = self.value(a).map(|x| -x);
        let grad = self.requires_grad(a);
        self.push(value, Op::Neg(a), grad)
    }

    pub fn exp(&mut self, a: Var) -> Var {
        let value = self.value(a).map(f64::exp);
        let grad = self.requires_grad(a);
        self.push(value, Op::Exp(a), grad)
    }

    pub fn ln(&mut self, a: Var) -> Var {
        let value = self.value(a).map(f64::ln);
        let grad = self.requires_grad(a);
        self.push(value, Op::Ln(a), grad)
    }

    pub fn ln_1p(&mut self, a: Var) -> Var {
        let value = self.value(a).map(f64::ln_1p);
        let grad = self.requires_grad(a);
        self.push(value, Op::Ln1p(a), grad)
    }

    pub fn sigmoid(&mut self, a: Var) -> Var {
        let value = self.value(a).map(special::sigmoid);
        let grad = self.requires_grad(a);
        self.push(value, Op::Sigmoid(a), grad)
    }

    pub fn softplus(&mut self, a: Var) -> Var {
        let value = self.value(a).map(special::softplus);
        let grad = self.requires_grad(a);
        self.push(value, Op::Softplus(a), grad)
    }

    pub fn gammaln(&mut self, a: Var) -> Var {
        let value = self.value(a).map(special::gammaln);
        let grad = self.requires_grad(a);
        self.push(value, Op::Gammaln(a), grad)
    }

    pub fn xlogy(&mut self, a: Var, b: Var) -> Var {
        let value = self
            .value(a)
            .binary(self.value(b), special::xlogy)
            .expect("shapes broadcast");
        let grad = self.binary_grad(a, b);
        self.push(value, Op::Xlogy(a, b), grad)
    }

    pub fn sum(&mut self, a: Var) -> Var {
        let value = Tensor::scalar(self.value(a).sum());
        let grad = self.requires_grad(a);
        self.push(value, Op::Sum(a), grad)
    }

    pub fn gather(&mut self, a: Var, map: GatherMap) -> Var {
        let parent = self.value(a);
        let data: Vec<f64> = map.map.iter().map(|&i| parent.data()[i]).collect();
        let value = Tensor::from_vec(map.out_shape.clone(), data);
        let grad = self.requires_grad(a);
        self.push(value, Op::Gather(a, map), grad)
    }

    /// Assemble a rank-1 vector of length `len` from scattered segments.
    /// Positions must be in-bounds; later writes win on overlap (JAX
    /// `.at[].set` chaining), though the IR validates disjointness upstream.
    pub fn scatter(&mut self, len: usize, parts: Vec<(Var, Vec<usize>)>) -> Var {
        let mut data = vec![0.0; len];
        let mut grad = false;
        for (var, positions) in &parts {
            let src = self.value(*var);
            assert_eq!(
                src.len(),
                positions.len(),
                "scatter segment length mismatch"
            );
            for (value, &pos) in src.data().iter().zip(positions.iter()) {
                data[pos] = *value;
            }
            grad |= self.requires_grad(*var);
        }
        self.push(
            Tensor::from_vec(vec![len], data),
            Op::Scatter { parts },
            grad,
        )
    }

    /// Elementwise select with broadcasting; gradient flows through the
    /// selected branch only (select semantics, not masked multiply, so an
    /// infinite adjoint cannot poison the unselected branch).
    pub fn where_select(&mut self, cond: Tensor, then_v: Var, else_v: Var) -> Var {
        let shape = Tensor::broadcast_shapes(cond.shape(), self.value(then_v).shape())
            .and_then(|s| Tensor::broadcast_shapes(&s, self.value(else_v).shape()))
            .expect("shapes broadcast");
        let cond_b = cond.broadcast_to(&shape).expect("cond broadcasts");
        let then_b = self
            .value(then_v)
            .broadcast_to(&shape)
            .expect("then broadcasts");
        let else_b = self
            .value(else_v)
            .broadcast_to(&shape)
            .expect("else broadcasts");
        let data: Vec<f64> = cond_b
            .data()
            .iter()
            .zip(then_b.data().iter().zip(else_b.data()))
            .map(|(&c, (&t, &e))| if c != 0.0 { t } else { e })
            .collect();
        let grad = self.binary_grad(then_v, else_v);
        self.push(
            Tensor::from_vec(shape, data),
            Op::Where {
                cond: cond_b,
                then_v,
                else_v,
            },
            grad,
        )
    }

    pub fn ordered_inverse(&mut self, a: Var) -> Var {
        let y = self.value(a);
        assert_eq!(y.rank(), 1, "Ordered constraint requires vector values");
        let mut data = Vec::with_capacity(y.len());
        let mut acc = y.data()[0];
        data.push(acc);
        for &yi in &y.data()[1..] {
            acc += yi.exp();
            data.push(acc);
        }
        let grad = self.requires_grad(a);
        let value = Tensor::from_vec(vec![y.len()], data);
        self.push(value, Op::OrderedInverse(a), grad)
    }

    /// Solve `L x = b`, L rank-2 lower-triangular, b rank-1.
    pub fn solve_lower(&mut self, l: Var, b: Var) -> Var {
        let l_t = self.value(l);
        let b_t = self.value(b);
        assert_eq!(l_t.rank(), 2);
        let n = l_t.shape()[0];
        assert_eq!(l_t.shape(), &[n, n]);
        assert_eq!(b_t.shape(), &[n]);
        let x = linalg::solve_lower(n, l_t.data(), b_t.data());
        let grad = self.binary_grad(l, b);
        self.push(Tensor::from_vec(vec![n], x), Op::SolveLower(l, b), grad)
    }

    /// Concatenate along the last axis; all parts share leading dims.
    pub fn concat_last(&mut self, parts: Vec<Var>) -> Var {
        assert!(!parts.is_empty());
        let lead = self.value(parts[0]).shape()[..self.value(parts[0]).rank() - 1].to_vec();
        let rows: usize = lead.iter().product();
        let mut last_total = 0usize;
        let mut grad = false;
        for &part in &parts {
            let t = self.value(part);
            assert!(t.rank() >= 1, "concat_last expects rank >= 1");
            assert_eq!(
                &t.shape()[..t.rank() - 1],
                lead.as_slice(),
                "leading dims differ"
            );
            last_total += t.shape()[t.rank() - 1];
            grad |= self.requires_grad(part);
        }
        let mut data = Vec::with_capacity(rows * last_total);
        for row in 0..rows {
            for &part in &parts {
                let t = self.value(part);
                let w = t.shape()[t.rank() - 1];
                data.extend_from_slice(&t.data()[row * w..(row + 1) * w]);
            }
        }
        let mut shape = lead;
        shape.push(last_total);
        let value = Tensor::from_vec(shape, data);
        self.push(value, Op::ConcatLast(parts), grad)
    }

    /// Materialize a broadcast of `a` to `shape`.
    pub fn broadcast(&mut self, a: Var, shape: &[usize]) -> Var {
        let value = self.value(a).broadcast_to(shape).expect("shape broadcasts");
        let grad = self.requires_grad(a);
        self.push(value, Op::Broadcast(a), grad)
    }

    pub fn reshape(&mut self, a: Var, shape: Vec<usize>) -> Var {
        let value = self.value(a).reshape(shape).expect("reshape size matches");
        let grad = self.requires_grad(a);
        self.push(value, Op::Reshape(a), grad)
    }

    /// Reverse pass from a scalar root; returns per-node adjoints for the
    /// requested leaves.
    pub fn backward(&self, root: Var, leaves: &[Var]) -> Vec<Tensor> {
        assert_eq!(self.value(root).len(), 1, "backward needs a scalar root");
        let mut adjoints: Vec<Option<Tensor>> = (0..self.nodes.len()).map(|_| None).collect();
        adjoints[root.0] = Some(Tensor::scalar(1.0));

        for id in (0..=root.0).rev() {
            if !self.nodes[id].requires_grad {
                continue;
            }
            let Some(adj) = adjoints[id].take() else {
                continue;
            };
            self.propagate(id, &adj, &mut adjoints);
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
            Op::Scatter { parts } => {
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
                if self.requires_grad(*then_v) {
                    let data: Vec<f64> = cond
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
                    let data: Vec<f64> = cond
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
            Op::Broadcast(a) => {
                self.accumulate(adjoints, *a, adj.clone());
            }
            Op::Reshape(a) => {
                let shape = self.value(*a).shape().to_vec();
                let grad = Tensor::from_vec(shape, adj.data().to_vec());
                self.accumulate(adjoints, *a, grad);
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
                let w = t.where_select(cond.clone(), v[0], v[1]);
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
        let cond = Tensor::from_vec(vec![2], vec![1.0, 1.0]);
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
}
