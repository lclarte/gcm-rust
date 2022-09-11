use std::f64::consts::PI;
use peroxide::numerical::*;

pub fn get_kappas_from_activation(activation : &String) -> (f64, f64) {
    return match activation.as_str() {
        "erf" => (2.0 / (3.0 * PI).sqrt(), 0.200364),
        "relu" => (0.5, ((PI - 2.0)/(4.0 * PI)).sqrt()),
        _ => panic!()
    };
}

pub fn get_additional_noise_variance_from_kappas(kappa1 : f64, kappastar : f64, gamma : f64) -> f64 {
    let kk1     = kappa1.powi(2);
    let kkstar  = kappastar.powi(2);

    let lambda_minus = (1.0 - gamma.sqrt()).powi(2);
    let lambda_plus       = (1.0 + gamma.sqrt()).powi(2);

    let integrand = |lambda : f64| -> f64 {((lambda_plus - lambda ) * (lambda - lambda_minus)).sqrt() / (kkstar + kk1 * lambda)};
    
    return 1.0 - kk1 * integral::integrate(integrand, (lambda_minus, lambda_plus), integral::Integral::G30K61(0.000001)) / (2.0 * PI);
}