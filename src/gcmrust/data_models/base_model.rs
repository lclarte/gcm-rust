
pub trait Partition {
    fn z0(&self, y : f64, w : f64, v : f64) -> f64;
    fn dz0(&self, y : f64, w : f64, v : f64) -> f64;
    fn f0(&self, y : f64, w : f64, v : f64) -> f64 {
        return self.z0(y, w, v) / self.dz0(y, w, v);
    }
}

pub trait ParameterPrior {
    fn get_gamma(&self) -> f64;
    fn get_rho(&self) -> f64;
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64);
}