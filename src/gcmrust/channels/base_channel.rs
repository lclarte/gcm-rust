pub trait Channel {
    /*
    Trait used for the estimators where we can compute f0, df0 (which is the case for the ERM) 
    without having to know what is the z0 of dz0.
    */
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64;
    fn df0(&self, y : f64, omega : f64, v : f64) -> f64;
    fn f0_square(&self, y : f64, omega : f64, v : f64) -> f64 {
        /*
        Why we use this function : it does not change anything for ERM estimator, 
        but for resampling estimators with weight p, we must not return f_g^2
        but the expectation E_p [ f_g(p) ] where e.g. f_g(p) = f_g if Bernoulli(p) = 1 and 0 otherwise
         */
        return self.f0(y, omega, v).powi(2);
    }
}

pub trait ChannelWithExplicitHatOverlapUpdate {
    fn update_hatoverlaps(&self, m : f64, q : f64, v : f64) -> (f64, f64, f64);
}