use peroxide::numerical::*;
use std::f64::consts::PI;

use crate::gcmrust::channels::base_channel;

static THRESHOLD_P : f64 = -10.0_f64;
static THRESHOLD_L : f64 = -10.0_f64;
static GK_PARAMETER : f64 = 0.000001;

pub fn p_out(x : f64, _beta : f64) -> f64 {
    if x > THRESHOLD_P {
        return (-1.0 as f64 + (-x).exp()).ln();
    }
    else {
        return -x;
    }
}

pub fn likelihood(x : f64, beta : f64) -> f64 {
    if x > THRESHOLD_L {
        return ( - beta * ( 1.0 as f64 + (-x).exp() ).ln() ).exp();
    }
    else {
        return ( beta * x).exp();
    }
}

pub fn z0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return   likelihood(local_field, beta) * (- z*z / 2.0).exp();
}

pub fn z0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64{
    if v > (10.0_f64).powi(-10) {
        let sqrt_v = v.sqrt();
        return integral::integrate(|z : f64| -> f64 {z0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
    }
    else {
        return   likelihood(y * w, beta);
    }
}

pub fn dz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    if local_field > THRESHOLD_L {
        return z *   likelihood(local_field, beta)  * (- z*z / 2.0).exp();
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
    return (z*z) *   likelihood(y * (z * sqrt_v + w), beta) * (-z*z / 2.0).exp() / (2.0 * PI).sqrt();
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
        
pub struct PseudoBayesLogistic {
    pub bound : f64,
    pub beta  : f64
}

impl base_channel::Channel for PseudoBayesLogistic{
    fn f0(&self, y : f64, w : f64, v : f64) -> f64 {
        let retour = dz0(y, w, v, self.beta, self.bound) / z0(y, w, v, self.beta, self.bound);
        if retour == std::f64::NAN {
            return 0.0;
        }
        return retour;
    }
    
    fn df0(&self, y : f64, w : f64, v : f64) -> f64 {
        let z0  = z0(y, w, v, self.beta, self.bound);
        let dz0 = dz0(y, w, v, self.beta, self.bound);
        let ddz0= ddz0(y, w, v, self.beta, self.bound,  Some(z0));
        let retour = ddz0 / z0 - (dz0 / z0).powi(2);
        if retour == std::f64::NAN {
            return 0.0;
        }
        return retour;
    }
}
