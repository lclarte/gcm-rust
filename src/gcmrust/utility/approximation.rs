use roots::{find_root_newton_raphson, SimpleConvergency};
use probability::{distribution::Gaussian, prelude::Inverse};
use peroxide::numerical::integral;

use crate::gcmrust::data_models::base_partition::Partition;
use crate::gcmrust::data_models::logit::Logit;
use crate::gcmrust::utility::constants::*;

pub fn approx_inverse_averaged_sigmoid(p : f64, variance : f64) -> f64 {
    let g = Gaussian::new(0.0, 1.0);
    return ( (1.0 / LOGIT_PROBIT_SCALING) + variance).sqrt() * g.inverse(p);
}

pub fn exact_inverse_averaged_sigmoid(p : f64, variance : f64) -> f64 {

    let model = Logit {
        noise_variance : 0.0
    };
    let start = approx_inverse_averaged_sigmoid(p, variance);

    let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter : 100 };

    let root = find_root_newton_raphson(start, |z : f64| -> f64 { p - model.z0(1.0, z, variance) }, |z : f64| -> f64 { - model.dz0(1.0, z, variance) }, &mut convergency);
    return match root {
        Err(e) => start,
        Ok(v )         => v,
    };
}

pub fn conditional_expectation_logit(m : f64, q : f64, delta_teacher : f64, rho : f64, student_local_field : f64) -> f64 {
    let conditional_mean = m / q * student_local_field;
    let conditional_variance = rho - m * m / q;

    let model = Logit {
        noise_variance : delta_teacher
    };

    return model.z0(1.0, conditional_mean, conditional_variance);
    
}