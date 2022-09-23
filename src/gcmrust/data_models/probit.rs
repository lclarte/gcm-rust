use statrs::function::*;
use std::f64::consts::PI;
use crate::gcmrust::data_models::base_partition;

pub struct Probit {
    pub noise_variance : f64
}

impl base_partition::Partition for Probit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        // consequence of these lines : REMOVE THE NOISE IN THE DEFINITION OF VSTAR !!!!!!!!
        let noisy_v = v + self.noise_variance;
        return 0.5 * erf::erfc(- (y * w) / (2.0 * noisy_v).sqrt());
    }
    
    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        // consequence of these lines : REMOVE THE NOISE IN THE DEFINITION OF VSTAR !!!!!!!!
        let noisy_v = v + self.noise_variance;
        return y * (- (w*w) / (2.0 * noisy_v)).exp() / (2.0 * PI * noisy_v).sqrt();
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        panic!("Not implemented yet !")
    }
}
