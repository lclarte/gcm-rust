use pyo3::prelude::*;

use crate::gcmrust::channels;
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
pub fn state_evolution(_py: Python, m: &PyModule) -> PyResult<()> {
    // Functions for classification
    m.add_function(wrap_pyfunction!(erm_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_state_evolution_gcm, m)?)?;
    
    m.add_function(wrap_pyfunction!(erm_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_state_evolution_matching, m)?)?;
    
    // Functions for regression
    
    m.add_function(wrap_pyfunction!(pseudo_bayes_ridge_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_ridge_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(erm_ridge_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(erm_ridge_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_regression_state_evolution_gcm, m)?)?;
    Ok(())
}

/// FUNCTIONS FOR STATE EVOLUTION

#[pyfunction]
fn erm_state_evolution_gcm(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64, f64, f64, f64) {
    let channel = channels::erm_logistic::ERMLogistic {
    };

    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    
    let noise_variance = delta + additional_variance;
    let prior = data_models::gcm::GCMPrior {
        teacher_norm : rho,
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        lambda : lambda_,
        rho : rho - additional_variance
    };

    let se = se::state_evolution::StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.01, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : false};

    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit { noise_variance : noise_variance };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition , &prior);
        
        return (m, q, v, mhat, qhat, vhat);
    }
    else {
        let data_model_partition = data_models::probit::Probit { noise_variance : noise_variance };
        
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }

}

#[pyfunction]
fn erm_state_evolution_matching(alpha : f64, delta : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let channel = channels::erm_logistic::ERMLogistic {
    };
    let prior = data_models::matching::MatchingPrior {
        lambda : lambda_,
        rho : rho
    };

    let se = se::state_evolution::StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

    // for logit and probit, delta is the variance of the noise in the likelihood
    // for the piecewise constant, delta is such that on [-delta, delta] the likelihood is 0.5
    // for the piecewise affine, delta is such that on [-delta, delta] the likelihood goes from 0 to 1
    let noise_variance = delta;
    let (m, q, v, mhat, qhat, vhat) : (f64, f64, f64, f64, f64, f64);

    // TODO : Learn Rust to simplify the code here
    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
    }
    else if data_model == "probit" {
        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
    }
    else if data_model == "piecewise_constant" {
        let data_model_partition = data_models::piecewise_constant::PiecewiseConstant {
            bound : delta
        };
        (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
    }
    else if data_model == "piecewise_affine" {
        let data_model_partition = data_models::piecewise_affine::PiecewiseAffine {
            bound : delta
        };
        (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
    }
    else {
        panic!("Not good data model!");
    }
    return (m, q, v, mhat, qhat, vhat);
}

#[pyfunction]
fn pseudo_bayes_state_evolution_gcm(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    // The rho factor comes from the scalar product between the teacher and the noise vector
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;
    
    if normalized {    
        let prior = data_models::gcm::GCMPriorPseudoBayes {
            teacher_norm : rho,
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };

        let se = se::state_evolution::StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

        if data_model == "logit" {
            let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }

        else {
            let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }
    }
    else {
        let prior = data_models::gcm::GCMPriorPseudoBayes {
            teacher_norm : rho,
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };
        let se =se::state_evolution::StateEvolution{ init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

        if data_model == "logit" {
            let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        } 
    }
}

#[pyfunction]
fn pseudo_bayes_state_evolution_matching(alpha : f64, beta : f64, delta : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let noise_variance = delta;
    let prior = data_models::matching::MatchingPriorPseudoBayes {
        beta_times_lambda : beta * lambda_,
        rho : rho
    };
    let se =se::state_evolution::StateEvolution{ init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

    if normalized {
        let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }
    }
    else {
        let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
            return (m, q, v, mhat, qhat, vhat);
        } 
    }
}

#[pyfunction]
fn bayes_optimal_state_evolution_gcm(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;

    let prior = data_models::gcm::GCMPriorBayesOptimal {
        teacher_norm : rho,
        gamma : gamma,
        rho : rho - additional_variance,
        kappa1 : kappa1,
        kappastar : kappastar
    };

    let se =se::state_evolution::StateEvolution{ init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

    let channel = data_models::logit::Logit {
        noise_variance : noise_variance
    };
    
    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }
    else {
        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }
}

#[pyfunction]
fn bayes_optimal_state_evolution_matching(alpha : f64, delta : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let noise_variance = delta;

    let prior = data_models::matching::MatchingPriorBayesOptimal {
        rho : rho
    };

    let se =se::state_evolution::StateEvolution{ init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };
    
    if data_model == "logit" {
        let channel = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }
    else {
        let channel = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        
        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }
}

/// FUNCTIONS FOR STATE EVOLUTION IN REGRESSION CASE

#[pyfunction]
fn pseudo_bayes_ridge_state_evolution_gcm(alpha : f64, delta_student : f64, delta_teacher : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    erm_ridge_state_evolution_gcm(alpha, delta_student, delta_teacher, gamma, kappa1, kappastar, lambda_, rho, se_tolerance, relative_tolerance, verbose)
}

#[pyfunction]
fn pseudo_bayes_ridge_state_evolution_matching(alpha : f64, delta_student : f64, delta_teacher : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    erm_ridge_state_evolution_matching(alpha, delta_student, delta_teacher, lambda_, rho, se_tolerance, relative_tolerance, verbose)
}

#[pyfunction]
fn bayes_optimal_regression_state_evolution_gcm(alpha : f64, delta_teacher : f64, gamma : f64, kappa1 : f64, kappastar : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);

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
        // gamma : gamma,
        rho : rho - additional_variance,
        student_noise_variance : delta_teacher + additional_variance,
        teacher_noise_variance : delta_teacher + additional_variance
    };

    let prior = data_models::gcm::GCMPriorBayesOptimal {
        teacher_norm : rho,
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        rho : rho - additional_variance
    };

    return se.state_evolution(alpha, &student_channel, &prior, true);
}

#[pyfunction]
fn erm_ridge_state_evolution_gcm(alpha : f64, delta_student : f64, delta_teacher : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64){
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    
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
        // gamma : gamma,
        rho : rho,
        student_noise_variance : delta_student,
        teacher_noise_variance : delta_teacher + additional_variance
    };

    /*
    let se = se::state_evolution::StateEvolution {
        init_m : 0.01, 
        init_q : 0.01,
        init_v : 0.99,
        se_tolerance : se_tolerance,
        relative_tolerance : relative_tolerance,
        verbose : verbose
    };

    let student_channel = channels::ridge_regression::RidgeChannel2 {
        noise_variance : delta_student
    };
    
    let teacher_model = data_models::gaussian::GaussianPartition {
        variance : delta_teacher + additional_variance
    };
    */

    let prior = data_models::gcm::GCMPrior {
        teacher_norm : rho,
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        // here there is no beta in the normalized case, so just lambda 
        lambda : lambda_,
        rho : rho - additional_variance
    };
    
    // return se.state_evolution(alpha, &student_channel, &teacher_model, &prior)
    return se.state_evolution(alpha, &student_channel, &prior, false);
}

#[pyfunction]
fn erm_ridge_state_evolution_matching(alpha : f64, delta_student : f64, delta_teacher : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64){
    
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
        rho : rho,
        student_noise_variance : delta_student,
        teacher_noise_variance : delta_teacher
    };

    let prior = data_models::matching::MatchingPrior {
        lambda : lambda_,
        rho : rho
    };

    return se.state_evolution(alpha, &student_channel, &prior, false);
}
