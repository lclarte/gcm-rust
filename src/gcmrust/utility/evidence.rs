use peroxide::numerical::integral;
use std::f64::consts::PI;

use crate::gcmrust::{data_models::{base_model::{ParameterPrior, Partition, PseudoBayesPrior}}, channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic};

static GK_PARAMETER : f64 = 0.000001;

pub fn psi_y(m : f64, q : f64, v : f64, student_partition : &impl Partition, true_model : &impl Partition, prior : &impl ParameterPrior) -> f64 {
    let vstar = prior.get_rho() - m * m / q;
    let mut somme = 0.0;
    let ys = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {true_model.z0(y, m / q.sqrt() * xi, vstar) * ( student_partition.z0(y, q.sqrt() * xi, v) ).ln() * (- xi * xi / 2.0) / (2.0 * PI).sqrt() },
            (-10.0, 10.0),
            integral::Integral::G30K61(GK_PARAMETER)
        );
    }

    return somme;
}

pub fn log_partition(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, student_partition : &NormalizedPseudoBayesLogistic, true_model : &impl Partition, prior : &(impl PseudoBayesPrior + ParameterPrior)) -> f64 {
    return prior.psi_w(mhat, qhat, vhat) + alpha * psi_y(m, q, v, student_partition, true_model, prior) - prior.get_gamma().sqrt() * m * mhat + 0.5 * (q * vhat - qhat * v) + 0.5 * v * vhat;
}

pub fn log_evidence(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, student_partition : &NormalizedPseudoBayesLogistic, true_model : &impl Partition, prior : &(impl PseudoBayesPrior + ParameterPrior)) -> f64 {
    // TODO : Possible that it's - 0.5 * (prior.beta_times_lambda).ln(), check the sign !!! 
    return 0.5 * (prior.get_prior_strength()).ln() + log_partition(m, q, v, mhat, qhat, vhat, alpha, student_partition, true_model, prior);
}