use std::time;
use gcmrust::gcmrust::*;

fn main() {
    let alpha = 1.0;
    let beta  = 1.0;
    let gamma = 1.0;
    let lamb  = 1.0;
    let (kappa1, kappastar) = utility::kappas::get_kappas_from_activation(&String::from("erf"));
    let additional_noise_var = utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma); 
    let rho : f64 = 1.0 - additional_noise_var;
    let delta = 0.5_f64.powi(2) + additional_noise_var;

    let data_model : String = String::from("probit");

    let (mut m, mut q, mut v) : (f64, f64, f64) = (0.01, 0.01, 1.0 - 0.001);

    let mut now : time::Instant;

    /*
    for i in 0..1000 {
        now = time::Instant::now();
        println!("Before iteration {}, m,q,v are {}, {}, {}", i, m, q, v);
        (m, q, v) = pseudo_bayes::state_evolution::iterate_se(m, q, v, alpha, beta, delta, gamma, kappa1, kappastar, lamb, rho, &data_model);
        println!("After iteration {}, m,q,v are {}, {}, {}", i, m, q, v);
        println!("Elapsed time {}", now.elapsed().as_micros());
    }
    println!("m, q, v are {}, {}, {}", m, q, v);
    */

    println!("Starting with ERM");

    (m, q, v) = (0.01, 0.01, 0.99);

    for i in 0..5 {
        now = time::Instant::now();
        println!("Before iteration {}, m,q,v are {}, {}, {}", i, m, q, v);
        (m, q, v) = erm::state_evolution::iterate_se(m, q, v, alpha, delta, gamma, kappa1, kappastar, lamb, rho);
        println!("After iteration {}, m,q,v are {}, {}, {}", i, m, q, v);
        println!("Elapsed time {}", now.elapsed().as_micros());
    }
}