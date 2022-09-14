use optimization::{Minimizer, GradientDescent, NumericalDifferentiation, Func};
use crate::gcmrust::channels::base_channel;

fn logistic_loss(z : f64) -> f64 {
    return (1.0 + (-z).exp()).ln();
}

fn logistic_loss_second_derivative(y : f64, z : f64) -> f64 {
    if (y * z).abs() > 500.0 {
        if y * z > 0.0 { return 1.0 / 4.0 * (-y * z).exp(); }
        else { return 1.0 / 4.0 * (y * z).exp(); }       
    }
    else {
        return 1.0 / (4.0 * (y * z / 2.0).cosh().powi(2));
    }
}

fn moreau_logistic_loss(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (x - omega).powi(2) / (2.0 * v) + logistic_loss(y*x);
}

/*
fn logistic_loss_derivative(y : f64, z : f64) -> f64 {
    if y * z > 0.0 {
        let x = (- y * z).exp();
        return - y * x / (1.0 + x);
    }
        
    else {
        return - y / ((y * z).exp() + 1.0);
    }
        
}

fn moreau_logistic_loss_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    // derivative with respect to x
    return (x - omega) / v + logistic_loss_derivative(y, x);
}
*/

fn proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    
    // Version of the code using third-party library
    let minimizer = GradientDescent::new();
    let to_minimize = NumericalDifferentiation::new(Func(|x: &[f64]| -> f64 {
        moreau_logistic_loss(x[0], y, omega, v)
    }));

    let x_sol = minimizer.minimize(&to_minimize, vec![omega - 20.0 * v, omega + 20.0 * v]);
    return x_sol.position[0];
    
}

pub struct ERMLogistic {

}

impl base_channel::Channel for ERMLogistic {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        return (lambda_star - omega) / v;
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        let dlambda_star = 1.0 / (1.0 + v * logistic_loss_second_derivative(y, lambda_star));
        return (dlambda_star - 1.0) / v;
    }
}