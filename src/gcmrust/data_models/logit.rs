use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use crate::gcmrust::{data_models::base_partition, channels::base_channel::Channel, utility::constants::*};

use super::base_partition::NormalizedChannel;


static LOGIT_QUAD_BOUND : f64 = INTEGRAL_BOUNDS; 
pub struct Logit {
    pub noise_variance : f64
}

fn noisy_sigmoid_likelihood(z : f64, noise_std : f64) -> f64 {
    // The exact version of the likelihood is a bit too slow, let's use an approximate form 
    // let integrand = |xi : f64| -> f64 { logistic::logistic( xi * noise_std + z ) * (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() };
    // return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
    return logistic::logistic( z / (1.0 + (LOGIT_PROBIT_SCALING * noise_std).powi(2) ).sqrt() );
}

impl base_partition::Partition for Logit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if self.noise_variance < 10.0_f64.powi(-10) {
            let integrand = |z : f64| -> f64 {logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
        }
        else {
            let integrand = |z : f64| -> f64 { noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
        }
        
    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if self.noise_variance < 10.0_f64.powi(-10) {
            let integrand = |z : f64| -> f64 { z * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()  };
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER))/ (2.0 * PI * v).sqrt();
        }
        else {
            let integrand = |z : f64| -> f64 { z * noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI * v).sqrt();
        }
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        if self.noise_variance < 10.0_f64.powi(-10) {
            let integrand = |z : f64| -> f64 { (z * z) * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()  };
            let integrale = integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER))/ (2.0 * PI * v).sqrt();
            let z0 = self.z0(y, w, v);
            return - z0 / v + integrale / v;
        }
        else {
            let integrand = |z : f64| -> f64 { (z * z) * noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
            let integrale = integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI * v).sqrt();
            let z0 = self.z0(y, w, v);
            return - z0 / v + integrale / v;
        }
    }

}

impl NormalizedChannel for Logit {
    
}