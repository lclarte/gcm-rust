use peroxide::prelude::PowOps;
use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use std::ops::RangeBounds;
use pyo3::prelude::*;

use crate::gcmrust::utility::constants::*;
use super::super::channels::base_channel::Channel;
use super::base_partition::{Partition, NormalizedChannel};

#[pyclass(unsendable)]
pub struct PiecewiseConstant {
    pub bound : f64
}

fn likelihood(z : f64, bound : f64) -> f64 {
    // The exact version of the likelihood is a bit too slow, let's use an approximate form 
    if z < -bound {
       0.0
    }
    else if z < bound {
        0.5
    }
    else {
        1.0
    }
}

impl Partition for PiecewiseConstant {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if y == 1.0 {
            // contribution of [-bound, bound] to Z
            let integral_pm_bound : f64 = 0.25 * (erf::erf((self.bound + w) / (2.0_f64 * v).sqrt()) - erf::erf((w - self.bound) / (2.0_f64 * v).sqrt()));
            // contribution of [bound, \infty] to Z
            let integral_bound_infty : f64 = 0.5 * (1.0 - erf::erf((self.bound - w) / (2.0_f64 * v).sqrt()));
            return integral_pm_bound + integral_bound_infty;
        }
        else {
            return 1.0 - self.z0(1.0, w, v);
        }
    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if y == 1.0 {
            return (2.0_f64.sqrt() * 0.25) / (PI* v).sqrt() * ((-(w + self.bound).powi(2) / (2.0 * v)).exp() - (-(w - self.bound).powi(2) / (2.0 * v)).exp()) + (- (w - self.bound).powi(2) / (2.0 * v)).exp() / (2.0 * PI * v).sqrt();
        }
        else {
            return - self.dz0(1.0, w, v)
        }
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        if y == 1.0 {
            0.25 * (2.0_f64 / PI).sqrt() / v.powf(3.0 / 2.0) * (- (-(w + self.bound).powi(2) / (2.0 * v)).exp() * (w + self.bound) + (-(w - self.bound).powi(2) / (2.0 * v)).exp() * (w - self.bound) ) + (- (w - self.bound).powi(2) / (2.0 * v)).exp() * (self.bound - w) / ((2.0 * PI).sqrt() * v.powf(3.0 / 2.0))
        }
        else {
            return - self.ddz0(1.0, w, v);
        }
    }

}

impl NormalizedChannel for PiecewiseConstant {
    
}

#[pymethods]
impl PiecewiseConstant {
    #[new]
    pub fn new(bound : f64) -> PyResult<Self>{
        Ok(PiecewiseConstant { bound : bound })
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