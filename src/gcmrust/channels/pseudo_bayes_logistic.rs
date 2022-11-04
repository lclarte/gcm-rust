use peroxide::numerical::*;
use std::f64::consts::PI;
use pyo3::prelude::*;
use statrs::function::*;

use crate::gcmrust::{data_models::base_partition::Partition, utility::constants::*};

static PSEUDO_BAYES_BOUND : f64 = INTEGRAL_BOUNDS;

pub fn likelihood(x : f64, beta : f64) -> f64 {
    return logistic::logistic(x).powf(beta);
}

pub fn z0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return likelihood(local_field, beta) * (- z*z / 2.0).exp() / (2.0 * PI).sqrt();
}

pub fn dz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return z * likelihood(local_field, beta) * (- z*z / 2.0).exp()  /  (2.0 * PI).sqrt() / sqrt_v;
}

fn ddz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    return (z*z) * likelihood(y * (z * sqrt_v + w), beta) * (-z*z / 2.0).exp() / (2.0 * PI).sqrt();
}


// Definition de la structure 

#[pyclass(unsendable)]
pub struct PseudoBayesLogistic {
    pub beta  : f64
}

impl PseudoBayesLogistic {
    fn integrate_function(&self, f : &dyn Fn(f64) -> f64) -> f64 {
        return integral::integrate(f, (-PSEUDO_BAYES_BOUND, PSEUDO_BAYES_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        // return integral::integrate(f, (-PSEUDO_BAYES_BOUND, PSEUDO_BAYES_BOUND), integral::Integral::GaussLegendre(16));
    }
}

impl Partition for PseudoBayesLogistic{
    fn z0(&self, y : f64, w : f64, v : f64) -> f64 {
        // return z0(y, w, v, self.beta, PSEUDO_BAYES_BOUND);
        if v > (10.0_f64).powi(-10) {
            let sqrt_v = v.sqrt();
            return self.integrate_function(&|z : f64| -> f64 {z0_integrand(z, y, w, sqrt_v, self.beta)}) ;
        }
    
        else {
            return likelihood(y * w, self.beta);
        }
    }

    fn dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        // return dz0(y, w, v, self.beta, PSEUDO_BAYES_BOUND);
        let sqrt_v = (v).sqrt();
    return self.integrate_function(&|z : f64| -> f64 {dz0_integrand(z, y, w, sqrt_v, self.beta)});
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        // let z0 = self.z0(y, w, v);
        // return ddz0(y, w, v, self.beta, PSEUDO_BAYES_BOUND, Some(z0));
        let sqrt_v = v.sqrt();
        let z0 : f64 = self.z0(y, w, v);

        let integrale = self.integrate_function(&| z : f64| -> f64 {ddz0_integrand(z, y, w, sqrt_v, self.beta)});
        return - z0 / v + integrale / v;
    }
}

// part for the Python

#[pymethods]
impl PseudoBayesLogistic {
    #[new]
    pub fn new(beta : f64) -> PyResult<Self>{
        Ok(PseudoBayesLogistic { beta : beta })
    }

    pub fn call_z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.z0(y, w, v);
    }

    pub fn call_dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.dz0(y, w, v);
    }

    pub fn call_ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.ddz0(y, w, v);
    }
}
