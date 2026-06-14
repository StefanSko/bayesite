//! Dense f64 tensors with NumPy/JAX broadcasting semantics.
//!
//! Only what the closed IR needs: elementwise ops with right-aligned
//! broadcasting, full reduction, and gather maps that mirror the indexing
//! subset validated by the Python compiler (tuples of full slices and
//! scalar/array indices, at most one array index, with NumPy's
//! move-to-front rule).

use crate::error::{Error, ErrorKind};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn mismatch(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataShapeMismatch, message)
}

impl Tensor {
    pub fn scalar(value: f64) -> Tensor {
        Tensor {
            shape: Vec::new(),
            data: vec![value],
        }
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape/data length mismatch"
        );
        Tensor { shape, data }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// The single value of a scalar (rank-0 or one-element) tensor.
    pub fn scalar_value(&self) -> Result<f64, Error> {
        if self.data.len() != 1 {
            return Err(mismatch(format!(
                "expected a scalar value, got shape {:?}",
                self.shape
            )));
        }
        Ok(self.data[0])
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Row-major strides for `shape`.
    pub fn strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// NumPy-style broadcast of two shapes (right-aligned).
    pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
        let rank = a.len().max(b.len());
        let mut out = vec![0usize; rank];
        for i in 0..rank {
            let da = if i < rank - a.len() {
                1
            } else {
                a[i - (rank - a.len())]
            };
            let db = if i < rank - b.len() {
                1
            } else {
                b[i - (rank - b.len())]
            };
            out[i] = if da == db || db == 1 {
                da
            } else if da == 1 {
                db
            } else {
                return Err(mismatch(format!("cannot broadcast shapes {a:?} and {b:?}")));
            };
        }
        Ok(out)
    }

    /// Value at `coords` under broadcasting from `self.shape` to a wider shape.
    fn broadcast_get(&self, out_coords: &[usize]) -> f64 {
        let offset = out_coords.len() - self.shape.len();
        let mut idx = 0usize;
        let mut stride = 1usize;
        for i in (0..self.shape.len()).rev() {
            let dim = self.shape[i];
            let coord = if dim == 1 { 0 } else { out_coords[offset + i] };
            idx += coord * stride;
            stride *= dim;
        }
        self.data[idx]
    }

    /// Elementwise binary op with broadcasting.
    pub fn binary(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, Error> {
        let shape = Tensor::broadcast_shapes(&self.shape, &other.shape)?;
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let mut coords = vec![0usize; shape.len()];
        for _ in 0..size {
            data.push(f(self.broadcast_get(&coords), other.broadcast_get(&coords)));
            for axis in (0..shape.len()).rev() {
                coords[axis] += 1;
                if coords[axis] < shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Ok(Tensor { shape, data })
    }

    /// Materialize this tensor broadcast to `shape`.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor, Error> {
        let target = Tensor::broadcast_shapes(&self.shape, shape)?;
        if target != shape {
            return Err(mismatch(format!(
                "cannot broadcast shape {:?} to {shape:?}",
                self.shape
            )));
        }
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let mut coords = vec![0usize; shape.len()];
        for _ in 0..size {
            data.push(self.broadcast_get(&coords));
            for axis in (0..shape.len()).rev() {
                coords[axis] += 1;
                if coords[axis] < shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Sum `self` down to `shape` (the adjoint of broadcasting to `self.shape`).
    pub fn reduce_to_shape(&self, shape: &[usize]) -> Tensor {
        if self.shape == shape {
            return self.clone();
        }
        let mut out = Tensor::zeros(shape);
        let offset = self.shape.len() - shape.len();
        let out_strides = Tensor::strides(shape);
        let mut coords = vec![0usize; self.shape.len()];
        for &v in &self.data {
            let mut idx = 0usize;
            for i in 0..shape.len() {
                let coord = if shape[i] == 1 { 0 } else { coords[offset + i] };
                idx += coord * out_strides[i];
            }
            out.data[idx] += v;
            for axis in (0..self.shape.len()).rev() {
                coords[axis] += 1;
                if coords[axis] < self.shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        out
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, Error> {
        if shape.iter().product::<usize>() != self.data.len() {
            return Err(mismatch(format!(
                "cannot reshape {:?} ({} elements) to {shape:?}",
                self.shape,
                self.data.len()
            )));
        }
        Ok(Tensor {
            shape,
            data: self.data.clone(),
        })
    }
}

/// One item of an evaluated index tuple.
#[derive(Debug, Clone)]
pub enum IndexAtom {
    Full,
    Scalar(i64),
    /// Integer index array with its own shape.
    Array {
        shape: Vec<usize>,
        values: Vec<i64>,
    },
}

/// A precomputed gather: `out[i] = base[map[i]]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GatherMap {
    pub out_shape: Vec<usize>,
    pub map: Vec<usize>,
}

fn wrap_index(index: i64, dim: usize, axis: usize) -> Result<usize, Error> {
    let dim_i = dim as i64;
    let wrapped = if index < 0 { index + dim_i } else { index };
    if wrapped < 0 || wrapped >= dim_i {
        return Err(mismatch(format!(
            "index {index} is out of bounds for axis {axis} with size {dim}"
        )));
    }
    Ok(wrapped as usize)
}

/// Build the gather map for `base[items...]` with NumPy/JAX semantics for
/// the subset the IR allows: at most one array index, scalar indices, and
/// full slices; the array-index block moves to the front when separated
/// from a scalar index by a slice.
pub fn gather_map(base_shape: &[usize], items: &[IndexAtom]) -> Result<GatherMap, Error> {
    if items.len() > base_shape.len() {
        return Err(mismatch(format!(
            "too many index items ({}) for expression with rank {}",
            items.len(),
            base_shape.len()
        )));
    }
    let array_positions: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, IndexAtom::Array { .. }))
        .map(|(i, _)| i)
        .collect();
    if array_positions.len() > 1 {
        return Err(mismatch(
            "index tuples support at most one non-scalar index expression",
        ));
    }

    // Pad with trailing full slices to the base rank.
    let mut padded: Vec<IndexAtom> = items.to_vec();
    while padded.len() < base_shape.len() {
        padded.push(IndexAtom::Full);
    }

    // NumPy move-to-front: the lone array block moves to the front of the
    // result when it is separated from a scalar index by a slice.
    let scalar_positions: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, IndexAtom::Scalar(_)))
        .map(|(i, _)| i)
        .collect();
    let moves_to_front = array_positions.first().is_some_and(|&array_pos| {
        scalar_positions.iter().any(|&scalar_pos| {
            let start = scalar_pos.min(array_pos) + 1;
            let stop = scalar_pos.max(array_pos);
            (start..stop).any(|p| matches!(items[p], IndexAtom::Full))
        })
    });

    // Output dims: a block per padded item (empty for scalars), in order,
    // with the array block optionally relocated to the front.
    enum OutBlock {
        BaseAxis(usize),
        ArrayDims(Vec<usize>),
    }
    let mut blocks: Vec<OutBlock> = Vec::new();
    for (axis, item) in padded.iter().enumerate() {
        match item {
            IndexAtom::Full => blocks.push(OutBlock::BaseAxis(axis)),
            IndexAtom::Scalar(_) => {}
            IndexAtom::Array { shape, .. } => blocks.push(OutBlock::ArrayDims(shape.clone())),
        }
    }
    if moves_to_front {
        if let Some(pos) = blocks
            .iter()
            .position(|b| matches!(b, OutBlock::ArrayDims(_)))
        {
            let block = blocks.remove(pos);
            blocks.insert(0, block);
        }
    }

    let mut out_shape: Vec<usize> = Vec::new();
    for block in &blocks {
        match block {
            OutBlock::BaseAxis(axis) => out_shape.push(base_shape[*axis]),
            OutBlock::ArrayDims(dims) => out_shape.extend(dims.iter().copied()),
        }
    }

    let base_strides = Tensor::strides(base_shape);
    let out_size: usize = out_shape.iter().product();
    let mut map = Vec::with_capacity(out_size);
    let mut coords = vec![0usize; out_shape.len()];
    for _ in 0..out_size {
        // Split output coords into per-block coords.
        let mut base_axis_coord = vec![0usize; base_shape.len()];
        let mut array_linear: Option<usize> = None;
        let mut cursor = 0usize;
        for block in &blocks {
            match block {
                OutBlock::BaseAxis(axis) => {
                    base_axis_coord[*axis] = coords[cursor];
                    cursor += 1;
                }
                OutBlock::ArrayDims(dims) => {
                    let mut linear = 0usize;
                    for &d in dims.iter() {
                        linear = linear * d + coords[cursor];
                        cursor += 1;
                    }
                    array_linear = Some(linear);
                }
            }
        }
        // Resolve scalar/array indices into base coordinates.
        let mut base_idx = 0usize;
        for (axis, item) in padded.iter().enumerate() {
            let coord = match item {
                IndexAtom::Full => base_axis_coord[axis],
                IndexAtom::Scalar(i) => wrap_index(*i, base_shape[axis], axis)?,
                IndexAtom::Array { values, .. } => {
                    let linear = array_linear.expect("array block present");
                    wrap_index(values[linear], base_shape[axis], axis)?
                }
            };
            base_idx += coord * base_strides[axis];
        }
        map.push(base_idx);
        for axis in (0..out_shape.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < out_shape[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Ok(GatherMap { out_shape, map })
}

/// Gather map slicing `[start, stop)` along the last axis.
pub fn slice_last_map(shape: &[usize], start: usize, stop: usize) -> GatherMap {
    let rank = shape.len();
    assert!(rank >= 1 && start <= stop && stop <= shape[rank - 1]);
    let width = shape[rank - 1];
    let rows: usize = shape[..rank - 1].iter().product();
    let mut map = Vec::with_capacity(rows * (stop - start));
    for row in 0..rows {
        for j in start..stop {
            map.push(row * width + j);
        }
    }
    let mut out_shape = shape[..rank - 1].to_vec();
    out_shape.push(stop - start);
    GatherMap { out_shape, map }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t1(values: &[f64]) -> Tensor {
        Tensor::from_vec(vec![values.len()], values.to_vec())
    }

    #[test]
    fn broadcasts_scalar_and_vector() {
        let a = Tensor::scalar(2.0);
        let b = t1(&[1.0, 2.0, 3.0]);
        let c = a.binary(&b, |x, y| x * y).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcasts_column_against_row() {
        // (2,1) op (3,) -> (2,3)
        let a = Tensor::from_vec(vec![2, 1], vec![10.0, 20.0]);
        let b = t1(&[1.0, 2.0, 3.0]);
        let c = a.binary(&b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn incompatible_shapes_error() {
        let a = t1(&[1.0, 2.0]);
        let b = t1(&[1.0, 2.0, 3.0]);
        assert!(a.binary(&b, |x, y| x + y).is_err());
    }

    #[test]
    fn reduce_to_shape_sums_broadcast_axes() {
        let c = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reduced = c.reduce_to_shape(&[3]);
        assert_eq!(reduced.data(), &[5.0, 7.0, 9.0]);
        let reduced_col = c.reduce_to_shape(&[2, 1]);
        assert_eq!(reduced_col.data(), &[6.0, 15.0]);
        let scalar = c.reduce_to_shape(&[]);
        assert_eq!(scalar.data(), &[21.0]);
    }

    #[test]
    fn gather_array_index_on_vector() {
        // base[idx] with idx = [2, 0, 1, 2]
        let map = gather_map(
            &[3],
            &[IndexAtom::Array {
                shape: vec![4],
                values: vec![2, 0, 1, 2],
            }],
        )
        .unwrap();
        assert_eq!(map.out_shape, vec![4]);
        assert_eq!(map.map, vec![2, 0, 1, 2]);
    }

    #[test]
    fn gather_scalar_index_drops_axis() {
        // base[1] on shape (2,3) -> shape (3,)
        let map = gather_map(&[2, 3], &[IndexAtom::Scalar(1)]).unwrap();
        assert_eq!(map.out_shape, vec![3]);
        assert_eq!(map.map, vec![3, 4, 5]);
    }

    #[test]
    fn gather_negative_index_wraps() {
        let map = gather_map(&[3], &[IndexAtom::Scalar(-1)]).unwrap();
        assert_eq!(map.out_shape, Vec::<usize>::new());
        assert_eq!(map.map, vec![2]);
    }

    #[test]
    fn gather_out_of_bounds_errors() {
        assert!(gather_map(&[3], &[IndexAtom::Scalar(3)]).is_err());
        assert!(gather_map(&[3], &[IndexAtom::Scalar(-4)]).is_err());
    }

    #[test]
    fn gather_tuple_slice_then_scalar() {
        // base[:, 1] on shape (2,3) -> (2,)
        let map = gather_map(&[2, 3], &[IndexAtom::Full, IndexAtom::Scalar(1)]).unwrap();
        assert_eq!(map.out_shape, vec![2]);
        assert_eq!(map.map, vec![1, 4]);
    }

    #[test]
    fn gather_array_moves_to_front_when_separated_by_slice() {
        // NumPy: base[idx, :, 0] keeps order (no slice between idx and 0)?
        // Positions: array at 0, scalar at 2, slice between -> moves to front,
        // but the array block is already first, so shape is (2, 4) either way.
        // The discriminating case: base[0, :, idx] on (5, 4, 3) ->
        // scalar at 0, slice at 1, array at 2 => idx dims move to the front:
        // result (2, 4) not (4, 2).
        let idx = IndexAtom::Array {
            shape: vec![2],
            values: vec![2, 0],
        };
        let map = gather_map(&[5, 4, 3], &[IndexAtom::Scalar(0), IndexAtom::Full, idx]).unwrap();
        assert_eq!(map.out_shape, vec![2, 4]);
        // out[i, j] = base[0, j, idx[i]] = idx[i] + 3*j
        assert_eq!(map.map[0], 2); // i=0 (idx 2), j=0
        assert_eq!(map.map[1], 2 + 3); // i=0, j=1
        assert_eq!(map.map[4], 0); // i=1 (idx 0), j=0
    }

    #[test]
    fn gather_adjacent_scalar_and_array_stays_in_place() {
        // base[0, idx] on (5, 3): no slice between -> in-position result (2,).
        let idx = IndexAtom::Array {
            shape: vec![2],
            values: vec![2, 0],
        };
        let map = gather_map(&[5, 3], &[IndexAtom::Scalar(0), idx]).unwrap();
        assert_eq!(map.out_shape, vec![2]);
        assert_eq!(map.map, vec![2, 0]);
    }

    #[test]
    fn rejects_two_array_indices() {
        let idx = IndexAtom::Array {
            shape: vec![2],
            values: vec![0, 1],
        };
        assert!(gather_map(&[3, 3], &[idx.clone(), idx]).is_err());
    }
}
