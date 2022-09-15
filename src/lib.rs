use core::panic;

use gcmrust::state_evolution::state_evolution::state_evolution;
use pyo3::prelude::*;

use gcmrust::data_models;
use gcmrust::channels;
use gcmrust::utility::kappas::get_additional_noise_variance_from_kappas;

pub mod gcmrust {
    pub mod state_evolution {
        pub mod integrals;
        pub mod state_evolution;
    }

    pub mod data_models {
        pub mod base_model;
        
        pub mod logit;
        pub mod probit;
        
        pub mod matching;
        pub mod gcm;
        
    }

    pub mod utility {
        pub mod errors;
        pub mod kappas;
    }

    pub mod channels {
        pub mod erm_logistic;
        pub mod pseudo_bayes_logistic;
        pub mod normalized_pseudo_bayes_logistic;
        pub mod base_channel;
    }

}

#[pyfunction]
fn erm_state_evolution_gcm(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64, f64, f64, f64) {
    let channel = channels::erm_logistic::ERMLogistic {
    };

    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    
    let noise_variance = delta + additional_variance;
    let prior = data_models::gcm::GCMPrior {
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        lambda : lambda_,
        rho : rho - additional_variance
    };

    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit { noise_variance : noise_variance };
        let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha, &channel, &data_model_partition , &prior, se_tolerance, relative_tolerance);
        
        return (m, q, v, mhat, qhat, vhat);
    }
    else {
        let data_model_partition = data_models::probit::Probit { noise_variance : noise_variance };
        
        let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
        return (m, q, v, mhat, qhat, vhat);
    }

}

#[pyfunction]
fn erm_state_evolution_matching(alpha : f64, delta : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64, f64, f64, f64) {
    let channel = channels::erm_logistic::ERMLogistic {
    };
    let prior = data_models::matching::Matching {
        lambda : lambda_,
        rho : rho
    };

    let noise_variance = delta;

    // TODO : Fix this shit with trait objects
    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha, &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
        return (m, q, v, mhat, qhat, vhat);
    }
    else if data_model == "probit" {
        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha, &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
        return (m, q, v, mhat, qhat, vhat);
    }    
    else {
        panic!("Not good data model!");
    }
}

#[pyfunction]
fn pseudo_bayes_state_evolution_gcm(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;
    let prior = data_models::gcm::GCMPriorPseudoBayes {
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        beta_times_lambda : beta * lambda_,
        // Give the TRUE prior norm, of the teacher, not of the student
        rho : rho - additional_variance
    };

    if normalized {
        let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
            bound : 10.0,
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha, &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
    }
    else {
        let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
            bound : 10.0,
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        } 
    }
}

#[pyfunction]
fn pseudo_bayes_state_evolution_matching(alpha : f64, beta : f64, delta : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool) -> (f64, f64, f64, f64, f64, f64) {
    let noise_variance = delta;
    let prior = data_models::matching::MatchingPseudoBayes {
        beta_times_lambda : beta *lambda_,
        rho : rho
    };

    if normalized {
        let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
            bound : 10.0,
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
    }
    else {
        let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
            bound : 10.0,
            beta  : beta
        };
    
        if data_model == "logit" {
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        }
        else {
            let data_model_partition = data_models::probit::Probit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
            return (m, q, v, mhat, qhat, vhat);
        } 
    }
}

/*
#[pyfunction]
fn bayes_optimal_state_evolution_gcm(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool) -> (f64, f64, f64, f64, f64, f64) {
    let noise_variance = delta;
    let prior = data_models::gcm::GCMBayesOptimalPrior {
        rho : rho,
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma
    };

    if data_model == "logit" {
        assert!(delta == 0.0);
        // TODO : Implement the logistic likelihood with arbitrary noise
        let channel = channels::pseudo_bayes_logistic::PseudoBayesLogistic {
            bound : 10.0,
            beta  : 1.0
        };

        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };

        let (m, q, v, mhat, qhat, vhat) = state_evolution(alpha,   &channel, &data_model_partition, &prior, se_tolerance, relative_tolerance);
        return (m, q, v, mhat, qhat, vhat);
    }
    else {
        // TODO _ IMPLEMENTER LE PROBIT LIKELIHOOD POUR LE PSEUDO BAYES (= JUSTE COPIER COLLER LE CODE DU DATA_MODEL)
        // TODO : SPLIT LE CHANNEL EN PARTITION + CHANNEL
        panic!("Not implemented");
    } 

}
*/

/* 
#[pyfunction]
fn pseudo_bayes_log_partition_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    return channels::normalized_pseudo_bayes_logistic::log_partition_matching(m, q, v, mhat, qhat, vhat, alpha, beta, delta, lambda,   &data_model);
}

#[pyfunction]
fn pseudo_bayes_log_partition_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    return channels::normalized_pseudo_bayes_logistic::log_partition_gcm(m, q, v, mhat, qhat, vhat, alpha, beta, delta, gamma, kappa1, kappastar, lambda,   &data_model);
}
*/

#[pyfunction]
fn test() {
    println!("The module is loaded correctly");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gcmpyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
 
    m.add_function(wrap_pyfunction!(erm_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(erm_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_matching, m)?)?;
    // m.add_function(wrap_pyfunction!(pseudo_bayes_log_partition_matching, m)?)?;
    // m.add_function(wrap_pyfunction!(pseudo_bayes_log_partition_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;

    Ok(())
}
