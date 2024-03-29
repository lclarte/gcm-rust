use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use pyo3::prelude::*;

use crate::gcmrust::utility::constants::*;
use super::super::channels::base_channel::Channel;
use super::base_partition::{Partition, NormalizedChannel};
use super::base_partition;

static LOGIT_QUAD_BOUND : f64 = 10.0; 

#[pyclass(unsendable)]
pub struct Logit {
    pub noise_variance : f64
}

fn noisy_sigmoid_likelihood(z : f64, noise_std : f64) -> f64 {
    // The exact version of the likelihood is a bit too slow, let's use an approximate form 
    // let integrand = |xi : f64| -> f64 { logistic::logistic( xi * noise_std + z ) * (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() };
    // return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
    // when noise_std = 0.0, normally it's the "correct" rigorous expression
    return logistic::logistic( z / (1.0 + (LOGIT_PROBIT_SCALING * noise_std).powi(2) ).sqrt() );
}

impl Partition for Logit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        let integrand = |z : f64| -> f64 { noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
        return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();    
    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        let integrand = |z : f64| -> f64 { z * noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
        return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI * v).sqrt();
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        let integrand = |z : f64| -> f64 { (z * z) * noisy_sigmoid_likelihood(y * (z * v.sqrt() + w), self.noise_variance.sqrt()) * (- z*z / 2.0).exp()};
        // below it was ... / (2.0 * PI).sqrt();
        let integrale = integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
        let z0 = self.z0(y, w, v);
        return - z0 / v + integrale / v;
        
    }

    fn get_output_type(&self) -> base_partition::OutputType {
        return base_partition::OutputType::BinaryClassification;
    }

}

impl Logit {
    pub fn squared_likelihood_expectation(&self, mean : f64, variance : f64) -> f64 {

        integral::integrate(
            |z : f64| -> f64 { noisy_sigmoid_likelihood(z * variance.sqrt() + mean, self.noise_variance.sqrt()).powi(2) * (- z*z / 2.0).exp() },
            (-INTEGRAL_BOUNDS, INTEGRAL_BOUNDS),
            integral::Integral::G30K61(GK_PARAMETER)
        ) / (2.0 * PI).sqrt()

    }
}

impl NormalizedChannel for Logit {
    
}


#[pymethods]
impl Logit {
    #[new]
    pub fn new(noise_variance : f64) -> PyResult<Self>{
        Ok(Logit { noise_variance : noise_variance })
    }

    fn call_z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.z0(y, w, v);
    }

    fn call_dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.dz0(y, w, v);
    }

    fn call_ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.ddz0(y, w, v);
    }

    fn call_f0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.f0(y, w, v);
    }

    fn call_df0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.df0(y, w, v);
    }
}