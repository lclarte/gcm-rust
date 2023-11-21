use probability::prelude::Gaussian;

use crate::gcmrust::{data_models::base_prior};
use crate::gcmrust::utility::constants::{INTEGRAL_BOUNDS, GK_PARAMETER};
use std::f64::consts::PI;
use peroxide::numerical::integral;

static INTEGRAL_BOUNDS_LASSO : f64 = 5.0;
static GK_PARAMETER_LASSO : f64 = 1e-5;

pub struct LassoPrior {
    pub lambda : f64,
    pub rho : f64
}

fn heaviside(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x == 0.0 {
        0.5
    } else {
        1.0
    }
}

fn fa(sigma: f64, r: f64, lambda_ : f64) -> f64 {
    let threshold = lambda_ * sigma * r.signum();
    (r - threshold) * heaviside(r.abs() - lambda_ * sigma)
}

fn fv(sigma: f64, r: f64, lambda_ : f64) -> f64 {
    // fv is sigma * partial_r fa
    sigma * heaviside(r.abs() - lambda_ * sigma)
}

fn fa_from_b_a(b : f64, a : f64, lambda_ : f64) -> f64 {
    fa(1.0 / a, b / a, lambda_)
}

fn partial_fa_from_b_a(b : f64, a : f64, lambda_ : f64) -> f64 {
    fv(1.0 / a, b / a, lambda_)
}



impl LassoPrior{
    //

    fn update_m(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let to_integrate = |wstar : f64| -> f64 {(-(wstar.abs())).exp() / 2.0 * 
            integral::integrate(
            |xi : f64| -> f64 { wstar * fa(1.0 / vhat , (mhat * wstar + qhat.sqrt() * xi) / vhat, self.lambda) * (-0.5 * xi.powi(2)).exp() / (2.0 * PI).sqrt() }, 
            (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO) )};
        
        integral::integrate(|wstar : f64| -> f64 { to_integrate(wstar) }, (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO))       
    }

    fn update_q(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let to_integrate = |wstar : f64| -> f64 {(-(wstar.abs())).exp() / 2.0 *  
            integral::integrate(
            |xi : f64| -> f64 { fa(1.0 / vhat , (mhat * wstar + qhat.sqrt() * xi) / vhat, self.lambda).powi(2) * (-0.5 * xi.powi(2)).exp() / (2.0 * PI).sqrt() }, 
            (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO) )};
        
        integral::integrate(|wstar : f64| -> f64 { to_integrate(wstar) }, (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO))       
    } 

    fn update_v(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        let to_integrate = |wstar : f64| -> f64 {(-(wstar.abs())).exp() / 2.0 *  
            integral::integrate(
            |xi : f64| -> f64 { fv(1.0 / vhat , (mhat * wstar + qhat.sqrt() * xi) / vhat, self.lambda) * (-0.5 * xi.powi(2)).exp() / (2.0 * PI).sqrt() }, 
            (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO) )};
        
        integral::integrate(|wstar : f64| -> f64 { to_integrate(wstar) }, (- INTEGRAL_BOUNDS_LASSO, INTEGRAL_BOUNDS_LASSO), integral::Integral::G30K61(GK_PARAMETER_LASSO))       
    }


}

impl base_prior::ParameterPrior for LassoPrior {
    fn get_gamma(&self) -> f64 {
        1.0
    }

    fn get_rho(&self) -> f64 {
        self.rho
    }

    //

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let m : f64 = self.update_m(mhat, qhat, vhat);
        let q : f64 = self.update_q(mhat, qhat, vhat);
        let v : f64 = self.update_v(mhat, qhat, vhat);

        return (m, q, v);
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        todo!()
    }
}