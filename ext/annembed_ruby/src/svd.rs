use magnus::{function, prelude::*, Error, Value, RArray, TryConvert};
use annembed::tools::svdapprox::{SvdApprox, RangeApproxMode, RangeRank, MatRepr};
use ndarray::Array2;

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let svd_module = parent.define_module("SVD")?;
    
    svd_module.define_singleton_method(
        "randomized_svd_rust",
        function!(randomized_svd, 3),
    )?;
    
    Ok(())
}

fn randomized_svd(matrix: Value, k: usize, n_iter: usize) -> Result<RArray, Error> {
    // Convert Ruby array to ndarray
    let rarray: RArray = TryConvert::try_convert(matrix)?;
    
    // Check if it's a 2D array
    let first_row: RArray = rarray.entry::<RArray>(0)?;
    let n_rows = rarray.len();
    let n_cols = first_row.len();
    
    if n_rows == 0 || n_cols == 0 {
        return Err(Error::new(
            magnus::exception::arg_error(),
            "Matrix cannot be empty",
        ));
    }
    
    if k > n_rows.min(n_cols) {
        return Err(Error::new(
            magnus::exception::arg_error(),
            format!("k ({}) cannot be larger than min(rows, cols) = {}", k, n_rows.min(n_cols)),
        ));
    }
    
    // Convert to ndarray Array2
    let mut matrix_data = Array2::<f64>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        let row: RArray = rarray.entry(i as isize)?;
        for j in 0..n_cols {
            let val: f64 = row.entry(j as isize)?;
            matrix_data[[i, j]] = val;
        }
    }
    
    // Create MatRepr for the full matrix
    let mat_repr = MatRepr::from_array2(matrix_data.clone());
    
    // Create SvdApprox instance
    let mut svd_approx = SvdApprox::new(&mat_repr);
    
    // Set up parameters for randomized SVD
    // Use RANK mode to specify the desired rank
    let params = RangeApproxMode::RANK(RangeRank::new(k, n_iter));
    
    // Perform SVD
    let svd_result = svd_approx.direct_svd(params)
        .map_err(|e| Error::new(magnus::exception::runtime_error(), e))?;
    
    // Extract U, S, V from the result - they are optional fields
    let u_matrix = svd_result.u.ok_or_else(|| {
        Error::new(magnus::exception::runtime_error(), "No U matrix in SVD result")
    })?;
    
    let s_values = svd_result.s.ok_or_else(|| {
        Error::new(magnus::exception::runtime_error(), "No S values in SVD result")
    })?;
    
    let vt_matrix = svd_result.vt.ok_or_else(|| {
        Error::new(magnus::exception::runtime_error(), "No V^T matrix in SVD result")
    })?;
    
    // Convert results to Ruby arrays
    // U matrix - convert ndarray to Ruby nested array
    let u_ruby = RArray::new();
    let u_shape = u_matrix.shape();
    for i in 0..u_shape[0] {
        let row = RArray::new();
        for j in 0..u_shape[1] {
            row.push(u_matrix[[i, j]])?;
        }
        u_ruby.push(row)?;
    }
    
    // S values - convert to Ruby array
    let s_ruby = RArray::new();
    for val in s_values.iter() {
        s_ruby.push(*val)?;
    }
    
    // V matrix (note: we have V^T, so we need to transpose)
    let v_ruby = RArray::new();
    let vt_shape = vt_matrix.shape();
    for i in 0..vt_shape[0] {
        let row = RArray::new();
        for j in 0..vt_shape[1] {
            row.push(vt_matrix[[i, j]])?;
        }
        v_ruby.push(row)?;
    }
    
    // Return [U, S, V^T] as a Ruby array
    let result = RArray::new();
    result.push(u_ruby)?;
    result.push(s_ruby)?;
    result.push(v_ruby)?;
    
    Ok(result)
}