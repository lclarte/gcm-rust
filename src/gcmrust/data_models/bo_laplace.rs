/*
bo_laplace.rs : when the prior is Laplace and matches the true prior (also Laplace, corresponding to the Bayesian case of Lasso)
NOTE that in the rest of the code, unless otherwise stated, the prior is always Gaussian (with norm 1)
*/

use probability::prelude::Gaussian;

use crate::gcmrust::{data_models::base_prior};
use std::f64::consts::PI;

pub struct BOLaplacePrior {
    pub gamma : f64,
    pub rho : f64
}


impl base_prior::ParameterPrior for BOLaplacePrior {
    fn get_gamma(&self) -> f64 {
        self.gamma
    }

    fn get_rho(&self) -> f64 {
        self.rho
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        /*
        We follow the computations of 2210.12760, equation (70)
        */
        todo!()
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        todo!()
    }
}