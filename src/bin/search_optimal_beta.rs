use simplers_optimization::Optimizer;
use std::io::Write;
use std::fs::OpenOptions;

use gcmrust::gcmrust::*;

fn test_opimisation() {
    let f = |x : &[f64]| -> f64 {(x[0] - 1.0).powi(2)};
    let x_interval = vec![(-10.0, 10.0)];
    let (value_opt, x_opt) = Optimizer::new(&f, &x_interval, true)
                                                .set_exploration_depth(5)
                                                .skip(50)
                                                .next().unwrap();

    println!("{}, {}", value_opt, x_opt[0]);
}

fn optimise_beta_for_error_probit_data(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, se_tolerance : f64, beta_min : f64, beta_max : f64) -> f64 {
    fn function_to_minimize(alpha : f64, beta : f64, delta : f64, gamma : f64, lambda : f64, rho : f64, kappa1 : f64, kappastar : f64, se_tolerance : f64) -> f64 {
        let (m, q, _v) = pseudo_bayes::state_evolution::state_evolution(alpha, beta, delta, gamma, kappa1, kappastar, lambda, rho, &String::from("probit"), se_tolerance, true);
        // let error = utility::errors::error_probit_model(m, q, rho, delta);
        let error = m / q.sqrt();
        println!("Trying for {}, error is {}", beta, error);
        return error;
    }

    let f = |beta:&[f64]| -> f64 { function_to_minimize(alpha, beta[0], delta, gamma, lambda, rho, kappa1, kappastar, se_tolerance)};
    let beta_interval      = vec![(beta_min, beta_max)];

    let (_value_opt, beta_opt) = Optimizer::minimize(&f, &beta_interval, 100);
    println!("Optimal value {}", beta_opt[0]);
    return beta_opt[0];
}

fn save_optimal_beta(filename : &String, noise_std : f64, activation : &String, n_over_p : f64, alpha_list : Vec<f64>, beta_opt_list : Vec<f64>) -> std::io::Result<()>  {
    let mut file = OpenOptions::new().write(true).append(true).open(filename).unwrap();
    let string = "alpha,gamma,noise_std,activation,beta_opt\n";
    file.write(string.as_bytes())?;
    for i in 0..alpha_list.len() {
        let (alpha, beta_opt) = (alpha_list[i], beta_opt_list[i]);
        let gamma = n_over_p / alpha;
        let string_array = vec![alpha.to_string(), gamma.to_string(), noise_std.to_string(), activation.to_owned(), beta_opt.to_string()];
        file.write(string_array.join(",").as_bytes())?;
        file.write(b"\n")?;
    }
    Ok(())
}

fn main() {
    
    let activation : String = String::from("erf");
    let se_tolerance      = 1e-3_f64;
    let noise_std         = 0.5_f64;
    let lambda = 1e-4_f64;
    let (beta_min, beta_max) = (1e-4_f64, 1.0_f64);
    let n_over_p               = 2.0;

    let alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0].to_vec();
    let mut beta_opt_list : Vec<f64> = Vec::new();

    for i in 0..alpha_list.len() {
        let alpha = alpha_list[i];
        println!("alpha = {}", alpha);
        let gamma                  = n_over_p / alpha;
        let (kappa1, kappastar) = utility::kappas::get_kappas_from_activation(&activation);
        let additional_noise_var = utility::kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, gamma);
        let delta = noise_std.powi(2) + additional_noise_var;
        let rho        = 1.0 - additional_noise_var;

        println!("Running for alpha = {}, gamma = {}, sigma = {}", alpha, gamma, noise_std);
        // let beta_opt = optimise_beta_for_error_probit_data(alpha, delta, gamma, kappa1, kappastar, lambda, rho, se_tolerance, beta_min, beta_max);
        test_opimisation();
        //println!("Optimal beta is {}", beta_opt);
        //beta_opt_list.push(beta_opt);
    }

    // let _result = save_optimal_beta(&String::from("beta_opt_error.csv"), noise_std, &activation, n_over_p, alpha_list, beta_opt_list);

}