use statrs::function::*;
use peroxide::numerical::integral;
use std::f64::consts::PI;
use crate::gcmrust::data_models::base_model;

static GK_PARAMETER : f64 = 0.000001;
static LOGIT_QUAD_BOUND : f64 = 10.0;
pub struct Logit {
    pub noise_variance : f64
}

impl base_model::Partition for Logit {
    fn z0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if self.noise_variance != 0.0 {
            panic!("For now, does not work if the noise variance is not 0 !");
        }
        let integrand = |z : f64| -> f64 {logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()};
        return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER)) / (2.0 * PI).sqrt();
    }

    fn dz0(&self, y : f64, w  : f64, v : f64) -> f64 {
        if v > (10.0_f64).powi(-10) {
            let integrand = |z : f64| -> f64 { z * logistic::logistic(y * (z * v.sqrt() + w)) * (- z*z / 2.0).exp()  };
            return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER))/ (2.0 * PI * v).sqrt();
        }
        else {
            return logistic::logistic(y * w);
        }
    }

    fn f0(&self, y : f64, w  : f64, v : f64) -> f64 {
        let result = self.dz0(y, w, v) / self.z0(y, w, v);
        return result;
    }

}
