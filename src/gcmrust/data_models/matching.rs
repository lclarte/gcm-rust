use crate::gcmrust::data_models::base_model;

pub struct Matching {
    pub lambda : f64,
    pub rho : f64
}

impl base_model::ParameterPrior for Matching {
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
}