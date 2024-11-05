use std::f64::consts::PI;
use crate::gcmrust::{channels::base_channel, data_models::base_partition::{Partition, self}};

/* 
Reminder : 
rho : norm of the teacher
alpha : sampling ratio (samples / students' dim)
gamma : student / teacher dimensions 
*/

pub struct RidgeChannel {
    /* 
    Consider a loss function 1/(2 noise_variance) (y - w^T x)^2
     */
    pub rho : f64,
    pub alpha : f64,
    // pub gamma : f64,
    pub student_noise_variance : f64,
    pub teacher_noise_variance : f64
}

// 

impl base_channel::ChannelWithExplicitHatOverlapUpdate for RidgeChannel {
    /*
        NOTE : Not sure of the expression 
    */
    
    fn update_hatoverlaps(&self, m : f64, q : f64, v : f64) -> (f64, f64, f64) {        
        // normalement : mhat est mult. par self.gamma.sqrt()
        // OLD VERSION : 
        // let mhat = self.alpha / (self.student_noise_variance + v);
        // let qhat = self.alpha * (self.rho + self.teacher_noise_variance + q - 2.0 * m ) / (self.student_noise_variance + v).powi(2);
        // let vhat = self.alpha / (self.student_noise_variance + v);

        let ratio : f64 = 1.0 / (self.student_noise_variance + v);
        
        let mhat = self.alpha * ratio;
        let qhat = self.alpha * ratio.powi(2) * (self.rho + self.teacher_noise_variance + q - 2.0 * m);
        let vhat = self.alpha * ratio;

        return (mhat, qhat, vhat);
    }
}

pub struct RidgeChannel2 {
    pub noise_variance : f64
}

impl base_channel::Channel for RidgeChannel2 {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        return (y - omega) / (v + self.noise_variance);
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        return - 1.0 / (v + self.noise_variance)
    }
}
