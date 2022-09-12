use peroxide::numerical::*;
use std::f64::consts::PI;

use crate::gcmrust::data_models;

static THRESHOLD_P : f64 = -10.0_f64;
static THRESHOLD_L : f64 = -10.0_f64;
static FT_QUAD_BOUND : f64 = 5.0;
static GK_PARAMETER : f64 = 0.00001;

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
        return integral::integrate(|z : f64| -> f64 {pseudobayes_z0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
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
    return integral::integrate(|z : f64| -> f64 {pseudobayes_dz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER)) /  (2.0 * PI * v).sqrt();
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

    let integrale = integral::integrate(| z : f64| -> f64 {pseudobayes_ddz0_integrand(z, y, w, sqrt_v, beta)}, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER));
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



pub fn logit_f0(y : f64, w  : f64, v : f64) -> f64 {
    let result = data_models::logit::dz0(y, w, v) / data_models::logit::z0(y, w, v);
    return result;
}

// 

pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64, beta : f64, data_model : &String) -> f64 {
    let mut somme : f64 = 0.0;
    let bound : f64 = FT_QUAD_BOUND;
    let teacher_dz0 : fn(f64, f64, f64) -> f64;
    
    if data_model == "logit" {
        teacher_dz0 = data_models::logit::dz0;
    }
    else {
        teacher_dz0 = data_models::probit::dz0;
    }
    let ys : [f64; 2] = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        let integrand = |xi : f64| -> f64 { (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() * pseudobayes_f0(y, (q).sqrt()*xi, v, beta, bound) * teacher_dz0(y, m / (q).sqrt() * xi, vstar) };
        somme = somme + integral::integrate(integrand, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER));
    }
    return somme;
}

pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64, beta : f64, data_model : &String) -> f64 {
    let mut somme : f64 = 0.0;
    let bound : f64 = FT_QUAD_BOUND;
    let teacher_z0 : fn(f64, f64, f64) -> f64;
    
    if data_model == "logit" {
        teacher_z0 = data_models::logit::z0;
    }
    else {
        teacher_z0 = data_models::probit::z0;
    }
    let ys : [f64; 2] = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        let integrand = |xi : f64| -> f64 { (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() * pseudobayes_f0(y, (q).sqrt()*xi, v, beta, bound).powi(2) * teacher_z0(y, m / (q).sqrt() * xi, vstar) };
        somme += integral::integrate(integrand, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER));
    }
    return somme;
}

pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64, beta : f64, data_model : &String) -> f64 {
    let mut somme : f64 = 0.0;
    let bound : f64 = FT_QUAD_BOUND;
    let teacher_z0 : fn(f64, f64, f64) -> f64;
    
    if data_model == "logit" {
        teacher_z0 = data_models::logit::z0;
    }
    else {
        teacher_z0 = data_models::probit::z0;
    }
    let ys : [f64; 2] = [-1.0, 1.0];

    for i in 0..2 {
        let y = ys[i];
        let integrand = |xi : f64| -> f64 { (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() * pseudobayes_df0(y, (q).sqrt()*xi, v, beta, bound) * teacher_z0(y, m / (q).sqrt() * xi, vstar) };
        somme = somme + integral::integrate(integrand, (-bound, bound), integral::Integral::G30K61(GK_PARAMETER));
    }
    return somme;
}