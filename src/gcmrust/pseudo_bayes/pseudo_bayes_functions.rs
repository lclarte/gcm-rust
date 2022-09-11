use peroxide::numerical::*;
use std::f64::consts::PI;
use statrs::function::*;

static THRESHOLD_P : f64 = -10.0_f64;
static THRESHOLD_L : f64 = -10.0_f64;

pub fn pseudobayes_p_out(x : f64, _beta : f64) -> f64 {
    if x > THRESHOLD_P {
        return (-1.0 as f64 + (-x).exp()).ln();
    }
    else {
        return -x;
    }
}

pub fn pseudobayes_likelihood(x : f64, beta : f64) -> f64 {
    if x > THRESHOLD_L {
        return ( - beta * ( 1.0 as f64 + (-x).exp() ).ln() ).exp();
    }
    else {
        return ( beta * x).exp();
    }
}

pub fn pseudobayes_z0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    return pseudobayes_likelihood(local_field, beta) * (- z*z / 2.0).exp();
}

pub fn pseudobayes_z0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64{
    if v > (10.0_f64).powi(-10) {
        let sqrt_v = v.sqrt();
        return integral::integrate(|z : f64| -> f64 {pseudobayes_z0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(1.0)) / (2.0 * PI).sqrt();
    }
    else {
        return pseudobayes_likelihood(y * w, beta);
    }
}

pub fn pseudobayes_dz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    let local_field : f64 = y * (z * sqrt_v + w);
    if local_field > THRESHOLD_L {
        return z * pseudobayes_likelihood(local_field, beta)  * (- z*z / 2.0).exp();
    }
    else {
        return z * (beta * local_field).exp() * (- z*z / 2.0).exp();
    }
}

fn pseudobayes_dz0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64{
    let sqrt_v = (v).sqrt();
    return integral::integrate(|z : f64| -> f64 {pseudobayes_dz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(1.0)) /  (2.0 * PI * v).sqrt();
}

fn pseudobayes_ddz0_integrand(z : f64, y : f64, w : f64, sqrt_v : f64, beta : f64) -> f64 {
    return (z*z) * pseudobayes_likelihood(y * (z * sqrt_v + w), beta) * (-z*z / 2.0).exp() / (2.0 * PI).sqrt();
}

pub fn pseudobayes_ddz0(y : f64, w : f64, v : f64, beta : f64, bound : f64, z0_option : Option<f64>) -> f64 {
    let sqrt_v = v.sqrt();
    let z0 : f64 = match z0_option {
        None        => pseudobayes_z0(y, w, sqrt_v * sqrt_v, beta, bound),
        Some(value ) => value
    };

    let integrale = integral::integrate(| z : f64| -> f64 {pseudobayes_ddz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(1.0));
    return - z0 / v + integrale / v;

}
        
pub fn pseudobayes_f0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64 {
    let retour = pseudobayes_dz0(y, w, v, beta, bound) / pseudobayes_z0(y, w, v, beta, bound);
    if retour == std::f64::NAN {
        return 0.0;
    }
    return retour;
}

pub fn pseudobayes_df0(y : f64, w : f64, v : f64, beta : f64, bound : f64) -> f64 {
    let z0  = pseudobayes_z0(y, w, v, beta, bound);
    let dz0 = pseudobayes_dz0(y, w, v, beta, bound);
    let ddz0= pseudobayes_ddz0(y, w, v, beta, bound,  Some(z0));
    let retour = ddz0 / z0 - (dz0 / z0).powi(2);
    if retour == std::f64::NAN {
        return 0.0;
    }
    return retour;
}

// Below, implement the probit and logit z0 and dz0 (needed to integrate mhat, qhat, vhat)


// ProbitDataModel:

pub fn probit_z0(y : f64, w  : f64, v : f64) -> f64 {
    return 0.5 * erf::erfc(- (y * w) / (2.0 * v).sqrt());
}

pub fn probit_dz0(y : f64, w  : f64, v : f64) -> f64 {
    return y * (- (w*w) / (2.0 * v)).exp() / (2.0 * PI * v).sqrt();
}

pub fn probit_f0(y : f64, w  : f64, v : f64) -> f64 {
    return probit_dz0(y, w, v) / probit_z0(y, w, v);
}

// LogisticDataModel:

pub fn logit_z0(y : f64, w  : f64, v : f64) -> f64 {
    let integrand = |z : f64| -> f64 {logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
    return integral::integrate(integrand, (-10.0, 10.0), integral::Integral::G30K61(1.0)) / (2.0 * PI).sqrt();

}

pub fn logit_dz0(y : f64, w  : f64, v : f64) -> f64 {
    if v > (10.0_f64).powi(-10) {
        let integrand = |z : f64| -> f64 { z * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
        return integral::integrate(integrand, (-10.0, 10.0), integral::Integral::G30K61(1.0)) / (2.0 * PI * v).sqrt();
    }
    else {
        return logistic::logistic(y * w);
    }
}

pub fn logit_f0(y : f64, w  : f64, v : f64) -> f64 {
    let result = logit_dz0(y, w, v) / logit_z0(y, w, v);
    return result;
}
