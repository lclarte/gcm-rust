use std::f64::consts::PI;
use crate::gcmrust::{channels::base_channel, data_models::base_partition::{Partition, self}};

pub struct RidgeChannel {
    pub rho : f64,
    pub alpha : f64,
    pub gamma : f64,
}

pub struct GaussianChannel {
    pub variance : f64
}

// 

impl base_channel::ChannelWithExplicitHatOverlapUpdate for RidgeChannel {
    fn update_hatoverlaps(&self, m : f64, q : f64, v : f64) -> (f64, f64, f64) {
        
        let mhat = (self.alpha * self.gamma.sqrt()) / (1.0 + v);
        let vhat =  self.alpha / (1.0 + v);
        let qhat = self.alpha * (self.rho + q - 2.0 * m ) / (1.0 + v).powi(2);

        return (mhat, qhat, vhat);
    }
}

impl base_partition::Partition for GaussianChannel {
    // normally, the noise_variance = 0.0 does not change anything 
    fn z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return (- 0.5 * (y - w).powi(2) / (self.variance + v)).exp() / (2.0 * PI * (self.variance + v)).sqrt();
    }

    fn dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return (y - w) / (v + self.variance) * (- 0.5 * (y - w).powi(2) / (self.variance + v)).exp() / (2.0 * PI * (self.variance + v)).sqrt();
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        let gaussienne = (- 0.5 * (y - w).powi(2) / (self.variance + v)).exp() / (2.0 * PI * (self.variance + v)).sqrt();
        return - 1.0 / (v + self.variance) * gaussienne + ((y - w) / (self.variance + v)).powi(2) * gaussienne;
    }
}