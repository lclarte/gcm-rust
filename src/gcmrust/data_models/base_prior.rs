// Prior, used for the penalization in ERM for example 

pub trait ParameterPrior {
    fn get_gamma(&self) -> f64;
    fn get_rho(&self) -> f64;
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64);
    fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> f64; 
}

pub trait PseudoBayesPrior {
    fn get_prior_strength(&self) -> f64;
}