/*
state_evolution_laplace.rs
All the functions where the teacher will come from the Laplace distribution
*/

use pyo3::prelude::*;

use crate::gcmrust::channels;
use crate::gcmrust::channels::base_channel;
use crate::gcmrust::utility;
use crate::gcmrust::data_models;
use crate::gcmrust::state_evolution as se;

/*
Notation : 
    - n: num of samples
    - p : parameters = student's dimsension
    - d : dimension = teachers' dimension 
    - alpha = n / p
    - gamma = p / d 
    - delta = variance of noiser 
    - kappa1, kappastar : property of the activation, cf. paper
    - rho : norm of the teacher \| wstar \|^2 (before the projection)

 */

#[pymodule]
pub fn state_evolution_laplace(_py: Python, m: &PyModule) -> PyResult<()> {
    // Functions for classification
    m.add_function(wrap_pyfunction!(erm_lasso_state_evolution_matching, m)?)?;
    Ok(())
}

/// FUNCTIONS FOR STATE EVOLUTION

#[pyfunction]
fn erm_lasso_state_evolution_matching(alpha : f64, delta_student : f64, delta_teacher : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64){
    /*
    In this function, the teacher is also coming from the Laplace distribution
     */
    let se = se::state_evolution::StateEvolutionExplicitOverlapUpdate {
        init_m : 0.01,
        init_q : 0.01, 
        init_v : 0.99,
        se_tolerance : se_tolerance,
        relative_tolerance : relative_tolerance,
        verbose : verbose
    };

    let student_channel = channels::ridge_regression::RidgeChannel {
        alpha : alpha,
        // gamma : 1.0,
        // NOTE : For now the rho factor is useless because the s.e. equation in matching_lasso assumes rho = 1
        rho : rho,
        student_noise_variance : delta_student,
        teacher_noise_variance : delta_teacher
    };

    let prior: data_models::matching_lasso::LassoPrior = data_models::matching_lasso::LassoPrior {
        lambda : lambda_,
        rho : rho
    };

    return se.state_evolution(alpha, &student_channel, &prior, false);
}