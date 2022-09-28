use core::panic;
use std::f32::consts::E;

use gcmrust::data_models::base_partition::Partition;
use gcmrust::state_evolution::state_evolution::{StateEvolution, StateEvolutionExplicitOverlapUpdate};
use gcmrust::utility;
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
        pub mod base_partition;
        pub mod base_prior;
        
        pub mod logit;
        pub mod probit;
        
        pub mod matching;
        pub mod gcm;
        
    }

    pub mod utility {
        pub mod errors;
        pub mod kappas;
        pub mod evidence;
        pub mod constants;
    }

    pub mod channels {
        pub mod erm_logistic;
        pub mod pseudo_bayes_logistic;
        pub mod normalized_pseudo_bayes_logistic;
        pub mod base_channel;
        pub mod ridge_regression;
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

    let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.01, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : false};

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

    let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

    let noise_variance = delta;

    if data_model == "logit" {
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }
    else if data_model == "probit" {
        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha, &channel, &data_model_partition, &prior);
        return (m, q, v, mhat, qhat, vhat);
    }    
    else {
        panic!("Not good data model!");
    }
}

#[pyfunction]
fn pseudo_bayes_state_evolution_gcm(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;
    
    if normalized {    
        let prior = data_models::gcm::GCMPriorPseudoBayes {
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };

        let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

        if data_model == "logit" {
            let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
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
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };
        let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

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
fn unstable_pseudo_bayes_projected_state_evolution_gcm(alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, data_model : String, se_tolerance : f64, relative_tolerance : bool, normalized : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;
    
    if normalized {    
        let prior = data_models::gcm::GCMPriorProjectedPseudoBayes {
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };

        let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

        if data_model == "logit" {
            let channel = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
                beta  : beta
            };
            let data_model_partition = data_models::logit::Logit {
                noise_variance : noise_variance
            };
            let (m, q, v, mhat, qhat, vhat) = se.state_evolution(alpha,   &channel, &data_model_partition, &prior);
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
        let prior = data_models::gcm::GCMPriorProjectedPseudoBayes {
            kappa1 : kappa1,
            kappastar : kappastar,
            gamma : gamma,
            beta_times_lambda : beta * lambda_,
            // Give the TRUE prior norm, of the teacher, not of the student
            rho : rho - additional_variance
        };
        let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

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
    let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

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
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;

    let prior = data_models::gcm::GCMPriorBayesOptimal {
        gamma : gamma,
        rho : rho - additional_variance,
        kappa1 : kappa1,
        kappastar : kappastar
    };

    let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

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

    let se = StateEvolution { init_m : 0.01, init_q : 0.01, init_v : 0.99, se_tolerance : se_tolerance, relative_tolerance : relative_tolerance, verbose : verbose };

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

//

#[pyfunction]
fn pseudo_bayes_ridge_state_evolution_gcm(alpha : f64, delta_student : f64, delta_teacher : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64) {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let se = StateEvolution {
        init_m : 0.01,
        init_q : 0.01, 
        init_v : 0.99,
        se_tolerance : se_tolerance,
        relative_tolerance : relative_tolerance,
        verbose : verbose
    };

    let teacher_channel = channels::ridge_regression::GaussianChannel {
        // shouldn't the teacher have 0 variance ? 
        variance : delta_teacher + additional_variance
    };

    let student_channel = channels::ridge_regression::GaussianChannel {
        variance : delta_student
    };

    let prior = data_models::gcm::GCMPriorPseudoBayes {
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        // here there is no beta in the normalized case, so just lambda 
        beta_times_lambda : lambda_,
        rho : rho - additional_variance
    };

    return se.state_evolution(alpha, &student_channel, &teacher_channel, &prior);
}

#[pyfunction]
fn erm_ridge_state_evolution_gcm(alpha : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda_ : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool, verbose : bool) -> (f64, f64, f64, f64, f64, f64){

    let se = StateEvolutionExplicitOverlapUpdate {
        init_m : 0.01,
        init_q : 0.01, 
        init_v : 0.99,
        se_tolerance : se_tolerance,
        relative_tolerance : relative_tolerance,
        verbose : verbose
    };

    let student_channel = channels::ridge_regression::RidgeChannel {
        alpha : alpha,
        gamma : gamma,
        rho : rho
    };

    let prior = data_models::gcm::GCMPrior {
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma,
        // here there is no beta in the normalized case, so just lambda 
        lambda : lambda_,
        rho
    };

    return se.state_evolution(alpha, &student_channel, &prior);
}


//

#[pyfunction]
fn pseudo_bayes_log_evidence_gcm
(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);

    let prior = data_models::gcm::GCMPriorPseudoBayes {
        rho : rho - additional_variance,
        beta_times_lambda : beta * lambda,
        gamma : gamma,
        kappa1 : kappa1,
        kappastar : kappastar
    };

    let student_partition = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
        beta  : beta
    };
    
    if data_model == "logit" {
        let true_model = data_models::logit::Logit {
            noise_variance : delta + additional_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
    else {
        let true_model = data_models::probit::Probit {
            noise_variance : delta + additional_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
}

#[pyfunction]
fn unstable_pseudo_bayes_projected_log_evidence_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);

    let prior = data_models::gcm::GCMPriorProjectedPseudoBayes {
        rho : rho - additional_variance,
        beta_times_lambda : beta * lambda,
        gamma : gamma,
        kappa1 : kappa1,
        kappastar : kappastar
    };

    let student_partition = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
        beta  : beta
    };
    
    if data_model == "logit" {
        let true_model = data_models::logit::Logit {
            noise_variance : delta + additional_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
    else {
        let true_model = data_models::probit::Probit {
            noise_variance : delta + additional_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
}

#[pyfunction]
fn pseudo_bayes_log_evidence_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let prior = data_models::matching::MatchingPriorPseudoBayes {
        rho : rho,
        beta_times_lambda : beta * lambda,
    };

    let student_partition = channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic {
        beta  : beta
    };
    
    if data_model == "logit" {
        let true_model = data_models::logit::Logit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
    else {
        let true_model = data_models::probit::Probit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
}

#[pyfunction]
fn bayes_optimal_log_evidence_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;

    let prior = data_models::gcm::GCMPriorBayesOptimal {
        gamma : gamma,
        rho : rho - additional_variance,
        kappa1 : kappa1,
        kappastar : kappastar
    };

    if data_model == "logit" {
        let channel = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        
        let data_model_partition = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &channel, &data_model_partition, &prior);
    }
    else {
        let channel = data_models::probit::Probit {
            noise_variance : noise_variance
        };

        let data_model_partition = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &channel, &data_model_partition, &prior);
    }
}

#[pyfunction]
fn bayes_optimal_log_evidence_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let prior = data_models::matching::MatchingPriorBayesOptimal {
        rho : rho,
    };

    if data_model == "logit" {
        let student_partition = data_models::logit::Logit {
            noise_variance : delta
        };

        let true_model = data_models::logit::Logit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
    else {

        let student_partition = data_models::logit::Logit {
            noise_variance : delta
        };

        let true_model = data_models::probit::Probit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &student_partition, &true_model, &prior);
    }
}

// 

// 

#[pyfunction]
fn test() {
    println!("The module is loaded correctly");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gcmpyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    
    // Functions for classification
    m.add_function(wrap_pyfunction!(erm_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(unstable_pseudo_bayes_projected_state_evolution_gcm, m)?)?;

    m.add_function(wrap_pyfunction!(erm_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_state_evolution_matching, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_state_evolution_matching, m)?)?;
    
    m.add_function(wrap_pyfunction!(pseudo_bayes_log_evidence_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(unstable_pseudo_bayes_projected_log_evidence_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_log_evidence_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_log_evidence_matching, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_log_evidence_matching, m)?)?;
    
    // Functions for regression

    m.add_function(wrap_pyfunction!(pseudo_bayes_ridge_state_evolution_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(erm_ridge_state_evolution_gcm, m)?)?;
    
    m.add_function(wrap_pyfunction!(test, m)?)?;

    Ok(())
}
