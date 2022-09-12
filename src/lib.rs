use pyo3::prelude::*;

use gcmrust::erm;
use gcmrust::pseudo_bayes;

pub mod gcmrust {
    pub mod pseudo_bayes {
        pub mod integrals;
        pub mod state_evolution;
    }

    pub mod erm {
        pub mod integrals;
        pub mod state_evolution;
    }

    pub mod data_models {
        pub mod logit;
        pub mod probit;
    }

    pub mod utility {
        pub mod errors;
        pub mod kappas;
    }
}

#[pyfunction]
fn erm_state_evolution_gcm_probit(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = erm::state_evolution::state_evolution_gcm_probit(alpha, delta, gamma, kappa1, kappastar, lambda_, rho, se_tolerance, relative_tolerance);
    return (m, q, v);
}

#[pyfunction]
fn erm_state_evolution_matching_probit(alpha : f64, delta : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = erm::state_evolution::state_evolution_matching_probit(alpha, delta, lambda_, rho, se_tolerance, relative_tolerance);
    return (m, q, v);
}

#[pyfunction]
fn erm_state_evolution_matching_logit(alpha : f64, delta : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = erm::state_evolution::state_evolution_matching_logit(alpha, delta, lambda_, rho, se_tolerance, relative_tolerance);
    return (m, q, v);
}

#[pyfunction]
fn pseudo_bayes_state_evolution_gcm_probit(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (m, q, v) = pseudo_bayes::state_evolution::state_evolution_gcm_probit(alpha, beta, delta, gamma, kappa1, kappastar, lambda, rho, &data_model , se_tolerance, relative_tolerance);
    return (m,q,v);
}

#[pyfunction]
fn test() {
    println!("The module is loaded correctly");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gcmpyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(erm_state_evolution_gcm_probit, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_gcm_probit, m)?)?;
    m.add_function(wrap_pyfunction!(erm_state_evolution_matching_probit, m)?)?;
    m.add_function(wrap_pyfunction!(erm_state_evolution_matching_logit, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;
    Ok(())
}
