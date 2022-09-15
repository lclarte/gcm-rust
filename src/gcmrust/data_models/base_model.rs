use crate::gcmrust::channels::base_channel::Channel;


pub trait Partition {
    /*
    Use notably for the teacher whose likelihood function generates a
    measure of probability through z0
    */
    fn z0(&self, y : f64, w : f64, v : f64) -> f64;
    fn dz0(&self, y : f64, w : f64, v : f64) -> f64;
    fn ddz0(&self, y : f64, w : f64, v : f64) -> f64;
}

/*
If we know z0 / dz0, we can compute f0 trivially because f0 = d_w(log(z0)) = z0 / dz0
*/
impl<T> Channel for T
where 
    T : Partition
    {
        fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
            return self.dz0(y, omega, v) / self.z0(y, omega, v);
        }

        fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
            let z0  = self.z0(y, omega, v);
            let dz0 = self.dz0(y, omega, v);
            let ddz0= self.ddz0(y, omega, v);
            let retour = (ddz0 / z0) - (dz0 / z0).powi(2);
            if retour == std::f64::NAN {
                return 0.0;
            }
            return retour;
        }
}

pub trait ParameterPrior {
    fn get_gamma(&self) -> f64;
    fn get_rho(&self) -> f64;
    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64);
    // fn psi_w(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64); 
}