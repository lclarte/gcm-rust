use probability::prelude::Gaussian;

use crate::gcmrust::{data_models::base_partition, channels};
use std::f64::consts::PI;

pub struct GaussianPartition {
    pub variance : f64
}

impl base_partition::Partition for GaussianPartition {
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

    fn get_output_type(&self) -> base_partition::OutputType {
        return base_partition::OutputType::Regression;
    }
}

pub struct GaussianChannel {
    pub variance : f64
}

impl channels::base_channel::Channel for GaussianChannel {

    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        return (y - omega) / (self.variance + v);
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        return - 1.0 / (self.variance + v);
    }
    
}