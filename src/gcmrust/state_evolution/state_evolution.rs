use crate::gcmrust::channels;
use crate::gcmrust::data_models;
use crate::gcmrust::state_evolution::integrals;


static MAX_ITER_ERM   : i16 = 100;
//

pub fn update_hatoverlaps(m : f64, q : f64, v : f64, alpha : f64, channel : &impl channels::base_channel::Channel, data_model : &impl data_models::base_model::Partition, prior : &impl data_models::base_model::ParameterPrior) -> (f64, f64, f64) {
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

pub fn iterate_se(m : f64, q : f64, v : f64, alpha : f64, channel : &impl channels::base_channel::Channel, data_model : &impl data_models::base_model::Partition, prior : &impl data_models::base_model::ParameterPrior) -> (f64, f64, f64, f64, f64, f64) {
    let (mhat, qhat, vhat);
    (mhat, qhat, vhat) = update_hatoverlaps(m, q, v, alpha, channel, data_model, prior);
    let (new_m, new_q, new_v) =  prior.update_overlaps(mhat, qhat, vhat);
    return (new_m, new_q, new_v, mhat, qhat, vhat);
}

pub fn state_evolution(alpha : f64, channel : &impl channels::base_channel::Channel, data_model : &impl data_models::base_model::Partition , prior : &impl data_models::base_model::ParameterPrior,  se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64, f64, f64, f64) {
    let (mut m, mut q, mut v) = (0.01, 0.01, 0.99);
    let (mut prev_m, mut prev_q, mut prev_v) : (f64, f64, f64);
    let (mut mhat, mut qhat, mut vhat) : (f64, f64, f64)= (0.0, 0.0, 0.0);

    let mut difference = 1.0;
    let mut counter : i16 = 0;

    while difference > se_tolerance && counter < MAX_ITER_ERM {
        (prev_m, prev_q, prev_v) = (m, q, v);
        (m, q, v, mhat, qhat, vhat) = iterate_se(m, q, v, alpha, channel, data_model, prior);
        if m == f64::NAN || q == f64::NAN || v == f64::NAN {
            panic!("One of the overlaps is NAN");
        }
        if relative_tolerance {
            difference = (m - prev_m).abs() / m.abs() + (q - prev_q).abs() / q.abs() + (v - prev_v).abs() / v.abs();
        }
        else {
            difference = (m - prev_m).abs() + (q - prev_q).abs() + (v - prev_v).abs();
        }
        counter += 1;
    }

    if counter == MAX_ITER_ERM {
        println!("Reached MAX_ITER_ERM in state evolution : last difference was {} / {}, relative tol. is {}", difference, se_tolerance, relative_tolerance);
    }

    return (m, q, v, mhat, qhat, vhat);
}
