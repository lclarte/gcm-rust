use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;

static GK_PARAMETER : f64 = 0.0001;

pub fn z0(y : f64, w  : f64, v : f64) -> f64 {
    let integrand = |z : f64| -> f64 {logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
    return integral::integrate(integrand, (-10.0, 10.0), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();

}

pub fn dz0(y : f64, w  : f64, v : f64) -> f64 {
    if v > (10.0_f64).powi(-10) {
        let integrand = |z : f64| -> f64 { z * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()  };
        return integral::integrate(integrand, (-10.0, 10.0), integral::Integral::G30K61(GK_PARAMETER))/ (2.0 * PI * v).sqrt();
    }
    else {
        return logistic::logistic(y * w);
    }
}
