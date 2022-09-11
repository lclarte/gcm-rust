use pyo3::prelude::*;

use gcmrust::erm;
use gcmrust::pseudo_bayes;

pub mod gcmrust {
    pub mod pseudo_bayes {
        pub mod pseudo_bayes_functions;
        pub mod state_evolution;
    }

    pub mod erm {
        pub mod integrals;
        pub mod state_evolution;
    }

    pub mod utility {
        pub mod errors;
        pub mod kappas;
    }
}

/*
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
*/

#[pyfunction]
fn erm_state_evolution(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = erm::state_evolution::state_evolution(alpha, delta, gamma, kappa1, kappastar, lambda_, rho, se_tolerance, relative_tolerance);
    return (m, q, v);
}

#[pyfunction]
fn pseudo_bayes_state_evolution(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = pseudo_bayes::state_evolution::state_evolution(alpha, beta, delta, gamma, kappa1, kappastar, lambda, rho, &data_model , se_tolerance, relative_tolerance);
    return (m,q,v);
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gcmpyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(erm_state_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution, m)?)?;
    Ok(())
}
