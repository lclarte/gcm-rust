use peroxide::prelude::PowOps;
use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use std::ops::RangeBounds;
use pyo3::prelude::*;

use crate::gcmrust::utility::constants::*;
use super::super::channels::base_channel::Channel;
use super::base_partition::{Partition, NormalizedChannel, self};

#[pyclass(unsendable)]
pub struct PiecewiseAffine {
    pub bound : f64
}

fn likelihood(z : f64, bound : f64) -> f64 {
    // The exact version of the likelihood is a bit too slow, let's use an approximate form 
    if z < -bound {
       0.0
    }
    else if z < bound {
        (z + bound) / (2.0 * bound)
    }
    else {
        1.0
    }
}

impl Partition for PiecewiseAffine {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {

        if y == 1.0 {
            let mut integral_pm_a = v.sqrt() / (2.0 * PI).sqrt() * (- (-(self.bound - w).powi(2) / (2.0 * v)).exp() + (- (self.bound + w).powi(2) / (2.0 * v)).exp() ) + ( erf::erf((self.bound + w) / ((2.0 * v).sqrt())) - erf::erf((w - self.bound) / ((2.0 * v).sqrt()))) * (self.bound + w) / 2.0;
            integral_pm_a   /= 2.0 * self.bound;
            let integral_a_infty = 0.5 * (1.0 - erf::erf((self.bound - w) / ((2.0 * v).sqrt())));
            integral_pm_a + integral_a_infty
        }
        else {
            // for y = -1.0
            1.0 - self.z0(1.0, w, v)
        }

    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {

        if y == 1.0 {
            return (-erf::erf((-self.bound + w)/(2.0 * v).sqrt()) + erf::erf((self.bound + w)/(2.0 * v).sqrt())) / (4.0 * self.bound);
        }
        else {
            // for y = -1.0
            return - self.dz0(1.0, w, v);
        }
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {

        if y == 1.0 {
            return -((-((self.bound + w).powi(2)/(2.0 * v))).exp() * (- 1.0 + ((2.0 * self.bound * w)/v).exp()))/(2.0 * self.bound * (2.0 * PI * v).sqrt());
        }
        else {
            // for y = -1.0
            return - self.ddz0(1.0, w, v);
        }

    }

    fn get_output_type(&self) -> base_partition::OutputType {
        return base_partition::OutputType::BinaryClassification;
    }
    

}

impl NormalizedChannel for PiecewiseAffine {
    
}

#[pymethods]
impl PiecewiseAffine {
    #[new]
    pub fn new(bound : f64) -> PyResult<Self>{
        Ok(PiecewiseAffine { bound : bound })
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