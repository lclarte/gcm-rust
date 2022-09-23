use crate::gcmrust::channels;
use crate::gcmrust::data_models;
use crate::gcmrust::data_models::base_partition;
use crate::gcmrust::data_models::base_prior::ParameterPrior;
use crate::gcmrust::data_models::base_partition::Partition;
use crate::gcmrust::channels::normalized_pseudo_bayes_logistic::NormalizedPseudoBayesLogistic;
use crate::gcmrust::data_models::base_prior::PseudoBayesPrior;
use crate::gcmrust::state_evolution::integrals;

static MAX_ITER_ERM   : i16 = 500;

//

pub struct StateEvolution {
    pub init_m : f64,
    pub init_q : f64, 
    pub init_v : f64,
    pub se_tolerance : f64,
    pub relative_tolerance : bool,
    pub verbose : bool
}

impl StateEvolution {
    pub fn state_evolution<C : channels::base_channel::Channel, P : Partition, Q : ParameterPrior>(&self, alpha : f64, channel : &C, data_model : &P, prior : &Q) -> (f64, f64, f64, f64, f64, f64) {
        let (mut m, mut q, mut v) = (self.init_m, self.init_q, self.init_v);
        let (mut prev_m, mut prev_q, mut prev_v) : (f64, f64, f64);
        let (mut mhat, mut qhat, mut vhat) : (f64, f64, f64)= (0.0, 0.0, 0.0);
        let (mut prev_mhat, mut prev_qhat, mut prev_vhat) : (f64, f64, f64);
    
        let mut difference = 1.0;
        let mut counter : i16 = 0;
    


        while difference > self.se_tolerance && counter < MAX_ITER_ERM {
            if self.verbose {
                println!("Iteration {} : m, q, v = {}, {}, {}", counter + 1, m, q,v);
            }
    
            (prev_m, prev_q, prev_v, prev_mhat, prev_qhat, prev_vhat) = (m, q, v, mhat, qhat, vhat);
            (m, q, v, mhat, qhat, vhat) = iterate_se(m, q, v, alpha, channel, data_model, prior);
    
            if m == f64::NAN || q == f64::NAN || v == f64::NAN {
                println!("One of the overlaps is NAN");
                return (prev_m, prev_q, prev_v, prev_mhat, prev_qhat, prev_vhat);
            }
            if self.relative_tolerance {
                difference = (m - prev_m).abs() / m.abs() + (q - prev_q).abs() / q.abs() + (v - prev_v).abs() / v.abs();
            }
            else {
                difference = (m - prev_m).abs() + (q - prev_q).abs() + (v - prev_v).abs();
            }

            counter += 1;
        }
    
        if counter == MAX_ITER_ERM {
            println!("Reached MAX_ITER_ERM in state evolution : last difference was {} / {}, relative tol. is {}", difference, self.se_tolerance, self.relative_tolerance);
        }
    
        return (m, q, v, mhat, qhat, vhat);
    }

    pub fn unstable_state_evolution_update_beta(&self, alpha : f64, channel : &NormalizedPseudoBayesLogistic, data_model : &impl base_partition::Partition, prior : &(impl PseudoBayesPrior + ParameterPrior)) -> (f64, f64, f64, f64, f64, f64) {
        // NOT TESTED FOR NOW !! 
        let (mut m, mut q, mut v) = (self.init_m, self.init_q, self.init_v);
        let (mut prev_m, mut prev_q, mut prev_v) : (f64, f64, f64);
        let (mut mhat, mut qhat, mut vhat) : (f64, f64, f64)= (0.0, 0.0, 0.0);
        let (mut prev_mhat, mut prev_qhat, mut prev_vhat) : (f64, f64, f64);
    
        let mut difference = 1.0;
        let mut counter : i16 = 0;
    
        // prior strength = beta * lambda
        let lambda = prior.get_prior_strength() / channel.beta;

        while difference > self.se_tolerance && counter < MAX_ITER_ERM {
            if self.verbose {
                println!("Iteration {} : m, q, v = {}, {}, {}", counter + 1, m, q,v);
            }
    
            (prev_m, prev_q, prev_v, prev_mhat, prev_qhat, prev_vhat) = (m, q, v, mhat, qhat, vhat);
            (m, q, v, mhat, qhat, vhat) = iterate_se(m, q, v, alpha, channel, data_model, prior);

            //update beta : minimize a function by iteration
            panic!("Not implemented yet !!");
    
            if m == f64::NAN || q == f64::NAN || v == f64::NAN {
                println!("One of the overlaps is NAN");
                return (prev_m, prev_q, prev_v, prev_mhat, prev_qhat, prev_vhat);
            }
            if self.relative_tolerance {
                difference = (m - prev_m).abs() / m.abs() + (q - prev_q).abs() / q.abs() + (v - prev_v).abs() / v.abs();
            }
            else {
                difference = (m - prev_m).abs() + (q - prev_q).abs() + (v - prev_v).abs();
            }

            counter += 1;
        }
    
        if counter == MAX_ITER_ERM {
            println!("Reached MAX_ITER_ERM in state evolution : last difference was {} / {}, relative tol. is {}", difference, self.se_tolerance, self.relative_tolerance);
        }
    
        return (m, q, v, mhat, qhat, vhat);
    }

}

pub fn update_hatoverlaps(m : f64, q : f64, v : f64, alpha : f64, channel : &impl channels::base_channel::Channel, data_model : &impl data_models::base_partition::Partition, prior : &impl ParameterPrior) -> (f64, f64, f64) {
    let vstar = prior.get_rho() - (m*m / q);

    let im = integrals::integrate_for_mhat(m, q, v, vstar, channel, data_model);
    let iq = integrals::integrate_for_qhat(m, q, v, vstar, channel, data_model);
    let iv = integrals::integrate_for_vhat(m, q, v, vstar, channel, data_model);

    let mhat = alpha *  prior.get_gamma().sqrt() * im;
    let vhat = - alpha * iv;
    let qhat = alpha * iq ;
    
    return (mhat, qhat, vhat);
}

//

pub fn iterate_se(m : f64, q : f64, v : f64, alpha : f64, channel : &impl channels::base_channel::Channel, data_model : &impl data_models::base_partition::Partition, prior : &impl ParameterPrior) -> (f64, f64, f64, f64, f64, f64) {
    let (mhat, qhat, vhat);
    (mhat, qhat, vhat) = update_hatoverlaps(m, q, v, alpha, channel, data_model, prior);
    let (new_m, new_q, new_v) =  prior.update_overlaps(mhat, qhat, vhat);
    return (new_m, new_q, new_v, mhat, qhat, vhat);
}

// 
