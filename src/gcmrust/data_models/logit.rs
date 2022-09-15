use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use crate::gcmrust::{data_models::base_model, channels::base_channel::Channel};

static GK_PARAMETER : f64 = 0.000001;
static LOGIT_QUAD_BOUND : f64 = 10.0; 
pub struct Logit {
    pub noise_variance : f64
}

fn noisy_sigmoid_likelihood(z : f64, noise_std : f64) -> f64 {
    let integrand = |xi : f64| -> f64 { logistic::logistic( xi * noise_std + z ) * (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() };
    return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
}

impl base_model::Partition for Logit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if self.noise_variance < 10.0_f64.powi(-10) {
            let integrand = |z : f64| -> f64 {logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
        }
        else {
            let integrand = |z : f64| -> f64 { noisy_sigmoid_likelihood(z, self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
        }
        
    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if v > (10.0_f64).powi(-10) {
            let integrand = |z : f64| -> f64 { z * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()  };
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER))/ (2.0 * PI * v).sqrt();
        }
        else {
            return logistic::logistic(y * w);
        }
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        panic!("Not implemented yet !");
    }

}
