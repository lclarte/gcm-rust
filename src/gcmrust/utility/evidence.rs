use peroxide::numerical::integral;
use std::f64::consts::PI;

use crate::gcmrust::{data_models::{base_prior::{ParameterPrior, PseudoBayesPrior}, base_partition::NormalizedChannel, matching::MatchingPrior, gcm::GCMPrior}, channels::{normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic, erm_logistic::{proximal_logistic_loss, logistic_loss}}, utility::constants::*};
use crate::gcmrust::data_models::base_partition::Partition;

use super::kappas::{marcenko_pastur_integral_without_zero, marcenko_pastur_integral};

pub fn ln_zero(x : f64) -> f64 {
    if x > 0.0 {
        x.ln()
    }
    else {
        0.0
    }
}

pub fn psi_y(m : f64, q : f64, v : f64, student_partition : &impl Partition, true_model : &impl Partition, prior : &impl ParameterPrior) -> f64 {
    let vstar = prior.get_rho() - (m * m / q);
    let mut somme = 0.0;
    let ys = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {true_model.z0(y, m / q.sqrt() * xi, vstar) * ln_zero(student_partition.z0(y, q.sqrt() * xi, v) ) * (- xi * xi / 2.0).exp() / (2.0 * PI).sqrt() },
            (-100.0, 100.0),
            integral::Integral::G30K61(0.0000000001)
        );
    }
    return somme;
}

pub fn unstable_psi_y(m : f64, q : f64, v : f64, student_partition : &impl Partition, true_model : &impl Partition, prior : &impl ParameterPrior) -> f64 {
    let vstar = prior.get_rho() - (m * m / q);
    let mut somme = 0.0;

    somme = somme + integral::integrate(
        |xi : f64| -> f64 { 
            let z0 = student_partition.z0(1.0, q.sqrt() * xi, v);
            (true_model.z0(1.0, m / q.sqrt() * xi, vstar) * ( z0 / (1.0 - z0) ).ln() + (1.0 - z0).ln() ) * (- xi * xi / 2.0).exp() / (2.0 * PI).sqrt()
        },
        (-20.0, 20.0), integral::Integral::G30K61(0.0000000001)
    );
    return somme;
}

pub fn log_partition(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, student_partition : &(impl NormalizedChannel + Partition), true_model : &impl Partition, prior : &(impl PseudoBayesPrior + ParameterPrior)) -> f64 {
    let psi_w_ = prior.psi_w(mhat, qhat, vhat);
    let psi_y_ = unstable_psi_y(m, q, v, student_partition, true_model, prior);
    // TODO : For the bayes optimal, we might need to divide by gamma somewhere (only true for the GCM)
    return psi_w_ + alpha * psi_y_ - m * mhat / prior.get_gamma().sqrt() + 0.5 * (q * vhat - qhat * v) + 0.5 * v * vhat;
}

pub fn log_evidence(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, student_partition : &(impl NormalizedChannel + Partition), true_model : &impl Partition, prior : &(impl PseudoBayesPrior + ParameterPrior)) -> f64 {
    // maybe remove the get_log_prior_strength() part, but in practice it doesn't seem to matter much
    return 0.5 * prior.get_log_prior_strength() + log_partition(m, q, v, mhat, qhat, vhat, alpha, student_partition, true_model, prior);
}

// compute here the evidence of Laplace 
// We'll need the training loss of Laplace, refer to this : https://github.com/IdePHICS/GCMProject/blob/dea00b06e9cb1118e46a756b6ea2d4a4665e9130/state_evolution/auxiliary/logistic_integrals.py#L93

// TODO : We need to integrate this for y = +1 and -1 to get the loss of the ERM 
// to get the Laplace we'll need the trace of the Hessian (see the paper for that).

fn integrand_training_loss_logistic(y : f64, xi : f64, m : f64, q : f64, v : f64, partition : &impl Partition, prior : &impl ParameterPrior) -> f64{
    /*
    Here we are only concerned with the training logistic loss, we will need to add the regularzation term (in this function or elsewere ?)
    the regularization term should be equal to (lambda_ / 2.0 * q)
    */
    let rho = prior.get_rho();
    let vstar = rho - (m * m / q);

    let omega : f64 = q.sqrt() * xi;
    let omegastar : f64 = m / q.sqrt() * xi;
    
    let x = proximal_logistic_loss(omega, v, y);
    let loss = logistic_loss(y * x);

    return (-xi * xi / 2.0).exp() / (2.0 * PI).sqrt() * partition.z0(y, omegastar, vstar) * loss;
}

pub fn training_loss_logistic(m : f64, q : f64, v : f64, partition : &impl Partition, prior : &impl ParameterPrior) -> f64 {
    // Return the training loss 1) without the regularization and 2) averaged over the training samples 
    // so that we get 1/n \sum_i ( loss for sample i )

    let mut somme = 0.0;
    let ys = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        somme = somme + integral::integrate(
            |xi : f64| -> f64 {integrand_training_loss_logistic(y, xi, m, q, v, partition, prior)},
            (-10.0, 10.0),
            integral::Integral::G30K61(0.0000000001)
        );
    }
    return somme
}

pub fn laplace_evidence_logistic_matching(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, partition : &impl Partition, prior : &MatchingPrior) -> f64 {
    // Function that returns log Z / p where p is the student's dimension
    // TODO : Test this function
    // first compute the training loss 

    let training_loss = training_loss_logistic(m, q, v, partition, prior);
    let lambda = prior.lambda;

    // we look at the determinant of the inverse Hessian ! 
    // we multiply training_loss by alpha because we divide log Z by p and not by n
    return (- alpha * training_loss - 0.5 * lambda * q) + 0.5 * (2.0 * PI).ln() + 0.5 * (1.0 / (vhat + lambda)).ln();
}

pub fn laplace_evidence_logistic_gcm(m : f64, q : f64, v : f64, mhat : f64, qhat : f64, vhat : f64, alpha : f64, partition : &impl Partition, prior : &GCMPrior) -> f64 {
    // still return log Z / p where p is the student's dimension

    let training_loss = training_loss_logistic(m, q, v, partition, prior);
    let lambda = prior.lambda;
    let gamma = prior.gamma;
    let kk1     = prior.kappa1.powi(2);
    let kkstar  = prior.kappastar.powi(2);

    let f = |s : f64| -> f64 {- (lambda + vhat * (kkstar + kk1 * s)).ln()};
    return (- alpha * training_loss - 0.5 * lambda * q) + 0.5 * (2.0 * PI).ln() + 0.5 * marcenko_pastur_integral(&f, gamma);
}