use roots::{SimpleConvergency, find_root_brent};
use crate::gcmrust::channels::base_channel;
use pyo3::prelude::*;

use super::base_channel::Channel;

static PROXIMAL_TOLERANCE : f64 = 0.001;

pub fn logistic_loss(z : f64) -> f64 {
    return (1.0 + (-z).exp()).ln();
}


fn logistic_loss_derivative(y : f64, z : f64) -> f64 {
    if y * z > 0.0 {
        let x = (- y * z).exp();
        return - y * x / (1.0 + x);
    }
        
    else {
        return - y / ((y * z).exp() + 1.0);
    }
        
}

fn logistic_loss_second_derivative(y : f64, x : f64) -> f64 {
    if (y * x).abs() > 500.0 {
        if y * x > 0.0 { return 1.0 / 4.0 * (-y * x).exp(); }
        else { return 1.0 / 4.0 * (y * x).exp(); }       
    }
    else {
        return 1.0 / (4.0 * (y * x / 2.0).cosh().powi(2));
    }
    /*    
    let expo = (x * y).exp();
    return 1.0 * expo / (1.0 + expo).powi(2) ;
    */
}

//

/*
fn moreau_logistic_loss(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (x - omega).powi(2) / (2.0 * v) + logistic_loss(y * x);
}

fn moreau_logistic_loss_second_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (1.0/ v) + logistic_loss_second_derivative(y, x);
}
 */

fn moreau_logistic_loss_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    // derivative with respect to x
    return (x - omega) / v + logistic_loss_derivative(y, x);
}

fn iterative_proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    let mut x = omega;
    for i in 0..100 {
        x = omega - v * logistic_loss_derivative(y, x);
    }
    return x;
}

pub fn proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    
    // USES BRENT

    let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter:30 };
    let root = find_root_brent(omega - 50.0 * v, omega + 50.0 * v, |x : f64| -> f64 {moreau_logistic_loss_derivative(x, y, omega, v)}, &mut convergency);
    return match root {
        Err(e) => iterative_proximal_logistic_loss(omega, v, y),
        Ok(v )         => v,
    };
    
}

// with weight to consider the resampling / subsampling, if weight = 1 we have the normal logistic regression 

fn moreau_logistic_loss_derivative_with_weight(x : f64, y : f64, omega : f64, v : f64, weight : f64) -> f64 {
    // derivative with respect to x
    return (x - omega) / v + weight * logistic_loss_derivative(y, x);
}

fn iterative_proximal_logistic_loss_with_weight(omega : f64, v : f64, y : f64, weight : f64) -> f64 {
    let mut x = omega;
    for i in 0..100 {
        x = omega - v * weight * logistic_loss_derivative(y, x);
    }
    return x;
}

pub fn proximal_logistic_loss_with_weight(omega : f64, v : f64, y : f64, weight : f64) -> f64 {
    let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter:30 };
    let root = find_root_brent(omega - 200.0 * v, omega + 200.0 * v, |x : f64| -> f64 {moreau_logistic_loss_derivative_with_weight(x, y, omega, v,  weight)}, &mut convergency);
    return match root {
        Err(e) => iterative_proximal_logistic_loss_with_weight(omega, v, y, weight),
        Ok(v )         => v,
    };
}

#[pyclass(unsendable)]
pub struct ERMLogistic {

}

impl base_channel::Channel for ERMLogistic {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        return (lambda_star - omega) / v;
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        let dlambda_star = 1.0 / (1.0 + v * logistic_loss_second_derivative(y, lambda_star));
        return (dlambda_star - 1.0) / v;
    }
}

#[pymethods]
impl ERMLogistic {
    #[new]
    pub fn new() -> PyResult<Self>{
        Ok(ERMLogistic {  })
    }

    fn call_f0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.f0(y, w, v);
    }

    fn call_df0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.df0(y, w, v);
    }
}

// Bernoulli resampling for the logistic channel


pub struct BernoulliSubsamplingLogistic {
    // define the proba of sampling 
    pub proba : f64,
}

impl base_channel::Channel for BernoulliSubsamplingLogistic{
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);

        // recall that we compute the expectation of f_g = - \partial_omega Moreau = (prox - omega) / v
        // but when p = 0 (no resample), f_g = 0 as prox = w
        return self.proba * (lambda_star - omega) / v ;
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        let dlambda_star = 1.0 / (1.0 + v * logistic_loss_second_derivative(y, lambda_star));
        return self.proba * (dlambda_star - 1.0) / v;
    }

    fn f0_square(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        return self.proba * ((lambda_star - omega) / v).powi(2);
    }
}

// Poisson resampling to model Bootstrap for the logistic channel

pub struct PoissonResamplingLogistric {
    // note in the default Bootstrap, the param is 1.0, but we can study other resamples 
    pub expected_resample : f64,
    // cutoff for the sum in the computation of expectations
    pub cutoff_expectation : i8
}

impl base_channel::Channel for PoissonResamplingLogistric {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        // Compute the expectation w.r.t the Poisson of f_g
        let mut sum : f64 = 0.0;
        // store the proba and update it from iteration to iteration to avoid overflows
        let mut proba : f64 = 1.0 / (self.expected_resample).exp();
        
        // loop from 0 to cutoff_expectation
        for weight in 0..self.cutoff_expectation {
            let lambda_star  = proximal_logistic_loss_with_weight(omega, v, y, weight as f64);
            sum += proba * (lambda_star - omega) / v;
            // after this line proba is p(i + 1)
            proba = proba * self.expected_resample / ((weight + 1) as f64);
        }

        return sum;
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let mut sum : f64 = 0.0;
        // store the proba and update it from iteration to iteration to avoid overflows
        let mut proba : f64 = 1.0 / (self.expected_resample).exp();
        
        // loop from 0 to cutoff_expectation
        for weight in 0..self.cutoff_expectation {
            let lambda_star  = proximal_logistic_loss_with_weight(omega, v, y, weight as f64);
            // don't forget the weight in the derivative of the logistic loss 
            let dlambda_star = 1.0 / (1.0 + v * (weight as f64) * logistic_loss_second_derivative(y, lambda_star));
            sum += proba * (dlambda_star - 1.0) / v;
            // after this line proba is p(i + 1)
            proba = proba * self.expected_resample / ((weight + 1) as f64);
        }

        return sum;
    }

    fn f0_square(&self, y : f64, omega : f64, v : f64) -> f64 {
        // Compute the expectation w.r.t the Poisson of f_g
        let mut sum : f64 = 0.0;
        // store the proba and update it from iteration to iteration to avoid overflows
        let mut proba : f64 = 1.0 / (self.expected_resample).exp();
        
        // loop from 0 to cutoff_expectation
        for weight in 0..self.cutoff_expectation {
            let lambda_star  = proximal_logistic_loss_with_weight(omega, v, y, weight as f64);
            sum += proba * ((lambda_star - omega) / v).powi(2);
            // after this line proba is p(i + 1)
            proba = proba * self.expected_resample / ((weight + 1) as f64);
        }

        return sum;
    }
}