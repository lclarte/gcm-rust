use peroxide::numerical::*;
use std::f64::consts::PI;

use crate::gcmrust::{channels::base_channel, data_models::{base_model::{Partition, ParameterPrior}, self}, utility};

static THRESHOLD_L : f64 = -10.0_f64;
static GK_PARAMETER : f64 = 0.000001;

pub fn likelihood(x : f64, beta : f64) -> f64 {
    return 1.0 / (1.0 + (- beta * x).exp());
}

pub fn z0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return  likelihood(local_field, beta) * (- z*z / 2.0).exp();
}

pub fn z0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64{
    if v > (10.0_f64).powi(-10) {
        let sqrt_v = v.sqrt();
        return integral::integrate(|z : f64| -> f64 {z0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
    }
    else {
        return  likelihood(y * w, beta);
    }
}

pub fn dz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    if local_field > THRESHOLD_L {
        return z *  likelihood(local_field, beta)  * (- z*z / 2.0).exp();
    }
    else {
        return z * (beta * local_field).exp() * (- z*z / 2.0).exp();
    }
}

fn dz0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64{
    let sqrt_v = (v).sqrt();
    return integral::integrate(|z : f64| -> f64 {dz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER)) /  (2.0 * PI * v).sqrt();
}

fn ddz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    return (z*z) *  likelihood(y * (z * sqrt_v + w), beta) * (-z*z / 2.0).exp() / (2.0 * PI).sqrt();
}

pub fn ddz0(y : f64, w : f64, v : f64, beta : f64, bound : f64, z0_option : Option<f64>) -> f64 {
    let sqrt_v = v.sqrt();
    let z0 : f64 = match z0_option {
        None        => z0(y, w, sqrt_v * sqrt_v, beta, bound),
        Some(value ) => value
    };

    let integrale = integral::integrate(| z : f64| -> f64 {ddz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER));
    return - z0 / v + integrale / v;

}
        
pub fn f0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64 {
    let retour = dz0(y, w, v, beta, bound) / z0(y, w, v, beta, bound);
    if retour == std::f64::NAN {
        return 0.0;
    }
    return retour;
}

pub fn df0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64 {
    let z0  = z0(y, w, v, beta, bound);
    let dz0 = dz0(y, w, v, beta, bound);
    let ddz0= ddz0(y, w, v, beta, bound,  Some(z0));
    let retour = ddz0 / z0 - (dz0 / z0).powi(2);
    if retour == std::f64::NAN {
        return 0.0;
    }
    return retour;

}

pub struct NormalizedPseudoBayesLogistic {
    pub bound : f64,
    pub beta  : f64
}

impl Partition for NormalizedPseudoBayesLogistic{
    // TODO : Trouver un moyen de pas avoir a recopier le code ebntre les deux PB
    fn z0(&self, y : f64, w : f64, v : f64) -> f64 {
        return z0(y, w, v, self.beta, self.bound);
    }

    fn dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return dz0(y, w, v, self.beta, self.bound);
    }

    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return ddz0(y, w, v, self.beta, self.bound, None);
    }
}
