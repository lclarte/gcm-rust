use statrs::function::*;

static PROX_TOLERANCE : f64 = 0.00000001;
static PROX_EPSILON : f64 = 0.1;
static MAX_ITER_PROX :  i16  = 1000;

fn logistic_loss(z : f64) -> f64 {
    return (1.0 + (-z).exp()).ln();
}

fn logistic_loss_derivative(z : f64) -> f64 {
    return - 1.0 / (1.0 + z.exp());
}

fn logistic_loss_second_derivative(z : f64) -> f64 {
    if (z).abs() > 500.0 {
        if z > 0.0{
            return 0.25 * (-z).exp();
        } 
        else{
            return 0.25 * z.exp();
        }
    } 
    else {
        return 1.0 / (4.0 * (z / 2.0).cosh().powi(2));
    } 
}

fn moreau_logistic_loss(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (x-omega).powi(2) / (2.0 * v) + logistic_loss(y*x);
}

fn moreau_logistic_loss_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    // derivative with respect to x
    return (x - omega) / v + y * logistic_loss_derivative(y * x);
}

fn proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    let mut x : f64     = omega;
    let mut old_x : f64 = 1.0;
    let mut counter : i16 = 0;
    while (old_x - x).abs() > PROX_TOLERANCE && counter < MAX_ITER_PROX {
        old_x = x;
        // do this for self-consistent equation ; x = - logistic_loss_derivative(y * x) * y * v + omega;
        x = x - PROX_EPSILON * moreau_logistic_loss_derivative(x, y, omega, v);
        counter += 1;
    }
    return x;
}

// define structure for proximal

fn f_mhat_plus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_plus = proximal_logistic_loss(omega, v, 1.0);
    return (-wstar.powi(2) / (2.0 * vstar)).exp() * (lambda_star_plus - omega);
}

fn f_mhat_minus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_minus = proximal_logistic_loss(omega, v, -1.0);
    return (-wstar.powi(2) / (2.0 * vstar)).exp() * (lambda_star_minus - omega);
}

// 

fn f_vhat_plus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_plus = proximal_logistic_loss(omega, v, 1.0);
    return (1.0 / (1.0 / v + 0.25 * (1.0 / (lambda_star_plus / 2.0).cosh().powi(2)))) * (1.0 + erf::erf(wstar/(2.0 * vstar).sqrt()));
}

fn f_vhat_minus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_minus = proximal_logistic_loss(omega, v, -1.0);
    return (1.0 / (1.0 / v + 0.25 * (1.0 / (- lambda_star_minus / 2.0).cosh().powi(2)))) * (1.0 - erf::erf(wstar/(2.0 * vstar).sqrt()));
}

// 

fn f_qhat_plus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_plus = proximal_logistic_loss(omega, v, 1.0);
    return (1.0 + erf::erf(wstar/(2.0*vstar).sqrt())) * (lambda_star_plus - omega).powi(2);
}

fn f_qhat_minus(xi : f64, m : f64, q : f64, v : f64, vstar : f64) -> f64  {
    let omega = (q).sqrt() * xi;
    let wstar = ( m / (q).sqrt()) * xi;
    let lambda_star_minus = proximal_logistic_loss(omega, v, -1.0);
    return (1.0 - erf::erf(wstar/(2.0*vstar).sqrt())) * (lambda_star_minus - omega).powi(2);
}
// 

pub mod probit_data_erm {

    use crate::gcmrust::erm::integrals::*;
    use std::f64::consts::PI;
    use peroxide::numerical::integral;

    static QUAD_BOUND : f64 = 10.0;
    static GK_PARAMETER : f64 = 0.0001;

    pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let i1 = integral::integrate(|xi : f64| -> f64 {f_vhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_vhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        return 0.5 * (i1 + i2);
    }
    
    pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let i1 = integral::integrate(|xi : f64| -> f64 {f_qhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_qhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        return 0.5 * (i1 + i2);
    }
    
    pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64) -> f64{
        let i1 = integral::integrate(|xi : f64| -> f64 {f_mhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_mhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
        return (i1 - i2) * (1.0 /(2.0 * PI * vstar).sqrt())
    }

}

// 

pub mod logistic_channel {
    use crate::gcmrust::erm::integrals::*;

    // define the z0, f0, etc. functions here

    pub fn f0(y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        return (lambda_star - omega) / v;
    }

    pub fn df0(y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        let dlambda_star = 1.0 / (1.0 + v * logistic_loss_second_derivative(lambda_star));
        return (dlambda_star - 1.0) / v;
    }

}

//

pub mod logit_data_erm {
    
    use peroxide::numerical::integral;
    use std::f64::consts::PI;
    
    use crate::gcmrust::data_models::logit;
    use crate::gcmrust::erm::integrals::*;

    static ERM_QUAD_BOUND : f64 = 10.0_f64;
    static GK_PARAMETER   : f64 = 0.0001_f64;

    pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let mut somme = 0.0_f64;
        let ys    = [-1.0, 1.0];

        for index in 0..2 {
            let y = ys[index];
            somme += integral::integrate(
                |xi : f64| -> f64 {(-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * logistic_channel::f0(y, q.sqrt() * xi, v) * logit::dz0(y, m / q.sqrt() * xi, vstar)}, 
                (- ERM_QUAD_BOUND, ERM_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)
            );
        }
        return somme;
    }
    
    pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let mut somme = 0.0_f64;
        let ys    = [-1.0, 1.0];

        for index in 0..2 {
            let y = ys[index];
            somme += integral::integrate(
                |xi : f64| -> f64 {(-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * logistic_channel::f0(y, q.sqrt() * xi, v).powi(2) * logit::z0(y, m / q.sqrt() * xi, vstar)}, 
                (- ERM_QUAD_BOUND, ERM_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)
            );
        }
        return somme;
    }
    
    pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64) -> f64{ 
        let mut somme = 0.0_f64;
        let ys    = [-1.0, 1.0];

        for index in 0..2 {
            let y = ys[index];
            somme += integral::integrate(
                |xi : f64| -> f64 {(-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt() * logistic_channel::df0(y, q.sqrt() * xi, v).powi(2) * logit::z0(y, m / q.sqrt() * xi, vstar)}, 
                (- ERM_QUAD_BOUND, ERM_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)
            );
        }
        return somme;
    }

}