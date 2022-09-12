use statrs::function::*;

static PROX_TOLERANCE : f64 = 0.00000001;
static PROX_EPSILON : f64 = 0.1;
static MAX_ITER_PROX :  i16  = 1000;

/*
fn logistic_loss(z : f64) -> f64 {
    return (1.0 + (-z).exp()).ln();
}
*/

fn logistic_loss_derivative(z : f64) -> f64 {
    return - 1.0 / (1.0 + z.exp());
}

/*
fn moreau_logistic_loss(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (x-omega).powi(2) / (2.0 * v) + logistic_loss(y*x);
}
*/

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

    pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let i1 = integral::integrate(|xi : f64| -> f64 {f_vhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_vhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        return 0.5 * (i1 + i2);
    }
    
    pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let i1 = integral::integrate(|xi : f64| -> f64 {f_qhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_qhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        return 0.5 * (i1 + i2);
    }
    
    pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64) -> f64{
        let i1 = integral::integrate(|xi : f64| -> f64 {f_mhat_plus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        let i2 = integral::integrate(|xi : f64| -> f64 {f_mhat_minus(xi, m, q, v, vstar) * (-xi*xi / 2.0).exp() / (2.0 * PI).sqrt()}, (-QUAD_BOUND, QUAD_BOUND), integral::Integral::G30K61(0.0001));
        return (i1 - i2) * (1.0 /(2.0 * PI * vstar).sqrt())
    }

}

//

pub mod logit_data_erm {
    use crate::gcmrust::erm::integrals::*;

    /*
    pub fn integrate_for_vhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
        let somme = 0.0_f64;
        for y in 
    }
    
    pub fn integrate_for_qhat(m : f64, q : f64, v : f64, vstar : f64) -> f64 { 
    
    }
    
    pub fn integrate_for_mhat(m : f64, q : f64, v : f64, vstar : f64) -> f64{
    
    }
    */

    /*
    def logistic_integrate_for_mhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * fout(y, np.sqrt(Q) * xi, V) * LogisticDataModel.dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def logistic_integrate_for_Qhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * fout(y, np.sqrt(Q)*xi, V)**2 * LogisticDataModel.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def logistic_integrate_for_Vhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * dfout(y, np.sqrt(Q)*xi, V) * LogisticDataModel.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme
    */

}