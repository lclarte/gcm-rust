use crate::gcmrust::data_models::base_prior;

use super::base_prior::PseudoBayesPrior;

pub struct MatchingPrior {
    pub lambda : f64,
    pub rho : f64
}

pub struct MatchingPriorPseudoBayes {
    pub beta_times_lambda : f64,
    pub rho : f64   
}

/*
pub struct MatchingBayesOptimal {
    pub lambda : f64,
    pub rho : f64
}
*/

impl base_prior::ParameterPrior for MatchingPrior {
    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64{
        return 1.0;
    }
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let v = 1. / (self.lambda + vhat);
        let q = (mhat.powi(2) + qhat) / (self.lambda + vhat).powi(2);
        let m = mhat / (self.lambda + vhat);
        return (m, q, v);
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        return - 0.5 * (self.lambda + vhat).ln() + 0.5 * (mhat * mhat + qhat) / (self.lambda + vhat);
    }
}

// 

impl base_prior::ParameterPrior for MatchingPriorPseudoBayes {
    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64{
        return 1.0;
    }
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let v = 1. / (self.beta_times_lambda + vhat);
        let q = (mhat.powi(2) + qhat) / (self.beta_times_lambda + vhat).powi(2);
        let m = mhat / (self.beta_times_lambda + vhat);
        return (m, q, v);
    }

    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64 {
        // return - 0.5 * np.log(beta_lambda + Vhat) + 0.5 * (mhat**2 + qhat) / (beta_lambda + Vhat)
        return - 0.5 * (self.beta_times_lambda + vhat).ln() + 0.5 * (mhat * mhat + qhat) / (self.beta_times_lambda + vhat);
    }
}

impl base_prior::PseudoBayesPrior for MatchingPriorPseudoBayes {
    fn get_prior_strength(&self) -> f64 {
        return self.beta_times_lambda;
    }
}

impl MatchingPriorPseudoBayes {
    pub fn derivative_psi_w_prior_strength(&self, mhat : f64, qhat : f64, vhat : f64) -> f64{
        return - 0.5 * (1. / (self.get_prior_strength() + vhat) + (mhat * mhat + qhat) / (self.get_prior_strength() + vhat).powi(2));
    }
}
