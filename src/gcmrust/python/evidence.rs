use pyo3::prelude::*;

use crate::gcmrust::channels;
use crate::gcmrust::utility;
use crate::gcmrust::data_models;

#[pymodule]
pub fn evidence(_py : Python, m : &PyModule)  -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pseudo_bayes_log_evidence_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_log_evidence_gcm, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_optimal_log_evidence_matching, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_bayes_log_evidence_matching, m)?)?;
    m.add_function(wrap_pyfunction!(erm_training_logistic_loss, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_evidence_logistic_matching, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_evidence_logistic_gcm, m)?)?;

    Ok(())
}

// FUNCTIONS TO COMPUTE THE EVIDENCE

#[pyfunction]
fn pseudo_bayes_log_evidence_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, beta : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);

    let prior = data_models::gcm::GCMPriorPseudoBayes {
        teacher_norm : rho,
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
    }}

#[pyfunction]
fn bayes_optimal_log_evidence_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    let noise_variance = delta + additional_variance;

    let prior = data_models::gcm::GCMPriorBayesOptimal {
        teacher_norm : rho,
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
fn bayes_optimal_log_evidence_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, delta : f64, rho : f64, data_model : String) -> f64 {
    let prior = data_models::matching::MatchingPriorBayesOptimal {
        rho : rho,
    };

    if data_model == "logit" {
        let channel = data_models::logit::Logit {
            noise_variance : delta
        };

        let true_model = data_models::logit::Logit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &channel, &true_model, &prior);
    }
    else if data_model == "probit" {

        let channel = data_models::logit::Logit {
            noise_variance : delta
        };

        let true_model = data_models::probit::Probit {
            noise_variance : delta
        };
        return utility::evidence::log_evidence(m, q, v, mhat, qhat, vhat, alpha, &channel, &true_model, &prior);
    }
    else {
        panic!("data_model must be either 'logit' or 'probit'");
    }
}

#[pyfunction]
fn erm_training_logistic_loss(m : f64, q : f64, v : f64, delta : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    
    let prior = data_models::matching::MatchingPrior {
        rho : rho,
        lambda : lambda
    };

    if data_model == "logit" {
    
        let true_model = data_models::logit::Logit {
            noise_variance : delta
        };
        return utility::evidence::training_loss_logistic(m, q, v, &true_model, &prior);
    }
    else if data_model == "probit" {

        let true_model = data_models::probit::Probit {
            noise_variance : delta
        };
        return utility::evidence::training_loss_logistic(m, q, v, &true_model, &prior);
    }

    else {
        panic!("data_model must be either 'logit' or 'probit'");
    }


}

#[pyfunction]
fn laplace_evidence_logistic_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, delta : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let prior = data_models::matching::MatchingPrior {
        rho : rho,
        lambda : lambda
    };

    if data_model == "logit" {
    
        let true_model = data_models::logit::Logit {
            noise_variance : delta
        };
        return utility::evidence::laplace_evidence_logistic_matching(m, q, v, mhat, qhat, vhat, alpha, &true_model, &prior);
    }
    else if data_model == "probit" {

        let true_model = data_models::probit::Probit {
            noise_variance : delta
        };
        return utility::evidence::laplace_evidence_logistic_matching(m, q, v, mhat, qhat, vhat, alpha, &true_model, &prior);
    }

    else {
        panic!("data_model must be either 'logit' or 'probit'");
    }
}

#[pyfunction]
fn laplace_evidence_logistic_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, data_model : String) -> f64 {
    let additional_variance = rho * utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
    // we don't add additional_variance to the noise as we want the training loss
    let noise_variance = delta;

    let prior = data_models::gcm::GCMPrior {
        rho : rho - additional_variance,
        lambda : lambda,
        teacher_norm : rho,
        kappa1 : kappa1,
        kappastar : kappastar,
        gamma : gamma
    };

    if data_model == "logit" {
    
        let true_model = data_models::logit::Logit {
            noise_variance : noise_variance
        };
        return utility::evidence::laplace_evidence_logistic_gcm(m, q, v, mhat, qhat, vhat, alpha, &true_model, &prior);
    }
    else if data_model == "probit" {

        let true_model = data_models::probit::Probit {
            noise_variance : noise_variance
        };
        return utility::evidence::laplace_evidence_logistic_gcm(m, q, v, mhat, qhat, vhat, alpha, &true_model, &prior);
    }

    else {
        panic!("data_model must be either 'logit' or 'probit'");
    }
}

