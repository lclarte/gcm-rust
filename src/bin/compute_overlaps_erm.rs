// For now, use erf activation, probit data model

use clap::Parser;
use std::env;
use std::fs::OpenOptions;
use std::fs::File;
use std::io::Write;

use gcmrust::gcmrust::utility::kappas;
use gcmrust::gcmrust::erm::state_evolution;

// defaut value is 1e-8
static SE_TOLERANCE : f64 = 0.00000001;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
   #[clap(short, long, value_parser)]
   alpha : f64,
   #[clap(short, long, value_parser)]
   noise_std: f64,
   #[clap(short, long, value_parser)]
   gamma : f64,
   #[clap(short, long, value_parser)]
    lambda : f64
}

fn save_overlaps(filename : &String, alpha : f64, noise_std : f64, activation : &String, gamma : f64, m : f64, q : f64, v : f64) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).append(true).open(filename).unwrap();
    let string_array = vec![alpha.to_string(), gamma.to_string(), noise_std.to_string(), activation.to_owned(), m.to_string(), q.to_string(), v.to_string()];
    let string_to_write = string_array.join(",");
    file.write(string_to_write.as_bytes())?;
    file.write(b"\n")?;
    Ok(())
}

fn main() {
    let args = Args::parse();

    let activation = String::from("erf");
    let (kappa1, kappastar) = kappas::get_kappas_from_activation(&activation);
    let additional_variance = kappas::get_additional_noise_variance_from_kappas(kappa1, kappastar, args.gamma);
    let delta = args.noise_std * args.noise_std + additional_variance;
    let rho = 1.0 - additional_variance;

    let (m, q, v) = state_evolution::state_evolution(args.alpha, delta, args.gamma, kappa1, kappastar, args.lambda, rho, SE_TOLERANCE, false);
    save_overlaps(&String::from("erm_overlaps.csv"), args.alpha, args.noise_std, &activation, args.gamma, m, q, v);
}