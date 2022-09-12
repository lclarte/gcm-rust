use statrs::function::*;
use std::f64::consts::PI;

use crate::gcmrust::data_models;

pub fn z0(y : f64, w  : f64, v : f64) -> f64 {
    return 0.5 * erf::erfc(- (y * w) / (2.0 * v).sqrt());
}

pub fn dz0(y : f64, w  : f64, v : f64) -> f64 {
    return y * (- (w*w) / (2.0 * v)).exp() / (2.0 * PI * v).sqrt();
}

pub fn f0(y : f64, w  : f64, v : f64) -> f64 {
    return data_models::probit::dz0(y, w, v) / data_models::probit::z0(y, w, v);
}
