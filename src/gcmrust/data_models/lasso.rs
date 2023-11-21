use probability::prelude::Gaussian;

use crate::gcmrust::{data_models::base_prior};
use std::f64::consts::PI;

pub struct LassoPrior {
    pub gamma : f64,
    pub rho : f64
}

impl base_prior::ParameterPrior for LassoPrior {
    fn get_gamma(&self) -> f64 {
        self.gamma
    }

    fn get_rho(&self) -> f64 {
        self.rho
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        todo!()
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        todo!()
    }
}