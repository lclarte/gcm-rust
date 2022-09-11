use crate::gcmrust::erm::integrals;

pub fn update_overlaps(mhat : f64, qhat : f64, vhat : f64, kappa1 : f64, kappastar : f64, gamma_arg : f64, lamb : f64) -> (f64, f64, f64) {
    let alpha  = gamma_arg;
    let gamma  = 1.0 / gamma_arg;
        
    let sigma  = kappa1;
    let kk     = kappastar * kappastar;
    let alphap = ( sigma * (1.0 + alpha.sqrt())).powi(2);
    let alpham = ( sigma * (1.0 - alpha.sqrt())).powi(2);

    if lamb == 0.0 {
        let den    = 1.0 + kk * vhat;
        let aux    = (((alphap+kk)*vhat+1.0)*((alpham+kk) * vhat + 1.0)).sqrt();
        let aux2   = (((alphap+kk)*vhat + 1.0) / ((alpham+kk) * vhat + 1.0)).sqrt();
        let mut iv = ((kk*vhat + 1.0) * ((alphap+alpham)*vhat + 2.0) - 2.0 *kk*vhat.powi(2) * (alphap*alpham).sqrt() -2.0 * aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+1.0)*sigma.powi(2));
        iv              = iv + 0.0_f64.max(1.0 - gamma)*kk/(1.0+vhat*kk);
        let i1     = (alphap * vhat*(-3.0 * den+aux)+4.0 * den * (-den+aux)+alpham*vhat*(-2.0 * alphap * vhat - 3.0 * den + aux))/(4.0 * alpha * vhat.powi(3) * sigma.powi(2)*aux);
        let i2     = (alphap * vhat+alpham*vhat*(1.0 - 2.0 * aux2) + 2.0 * den * (1.0 - aux2))/(4.0 * alpha * vhat.powi(2) * aux * sigma.powi(2));
        let i3     = (2.0 * vhat * alphap*alpham+(alphap+alpham) * den- 2.0 * (alphap*alpham).sqrt() * aux)/(4.0 * alpha * den.powi(2) * sigma.powi(2) * aux);
        let mut iq = (qhat + mhat.powi(2)) * i1 + (2.0*qhat+mhat.powi(2)) * kk * i2 + qhat * kk.powi(2) * i3;
        iq              = iq + 0.0_f64.max(1.0-gamma)*qhat*(kk / den).powi(2);
        let im     = ((alpham + alphap+2.0*kk)*vhat+2.0 - 2.0 * aux)/(4.0*alpha*(vhat/sigma).powi(2));
        let v = iv;
        let m = mhat * (gamma_arg).sqrt() * im;
        let q = iq;
        return (m, q, v);
    }
    else {
        let den    = lamb+kk*vhat;
        let aux    = (((alphap+kk)*vhat+lamb)*((alpham+kk)*vhat+lamb)).sqrt();
        let aux2   = (((alphap+kk)*vhat+lamb)/((alpham+kk)*vhat+lamb)).sqrt();
        let mut iv = ((kk*vhat+lamb)*((alphap+alpham)*vhat+2.0 * lamb)-2.0 * kk*vhat.powi(2)*(alphap*alpham).sqrt()-2.0 * lamb*aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+lamb)*sigma.powi(2));
        iv              = iv + f64::max(0.0, 1.0-gamma)*kk/(lamb+vhat*kk);
        let i1     = (alphap*vhat*(-3.0*den+aux)+4.0*den*(-den+aux)+alpham*vhat*(-2.0*alphap*vhat-3.0*den+aux))/(4.0*alpha*vhat.powi(3)*sigma.powi(2)*aux);
        let i2     = (alphap*vhat+alpham*vhat*(1.0-2.0*aux2)+2.0*den*(1.0-aux2))/(4.0*alpha*vhat.powi(2)*aux*sigma.powi(2));
        let i3     = (2.0*vhat*alphap*alpham+(alphap+alpham)*den-2.0*(alphap*alpham).sqrt()*aux)/(4.0*alpha*den.powi(2)*sigma.powi(2)*aux);
        let mut iq = (qhat+mhat.powi(2))*i1+(2.0*qhat+mhat.powi(2))*kk*i2+qhat*kk.powi(2)*i3;
        iq              = iq + f64::max(0.0, 1.0 - gamma)*qhat*kk.powi(2)/den.powi(2);
        let im     = ((alpham+alphap+2.0*kk)*vhat+2.0*lamb-2.0*aux)/(4.0*alpha*vhat.powi(2)*sigma.powi(2));


        let v = iv;
        let m = mhat * (gamma_arg).sqrt() * im;
        let q = iq;
        return (m, q, v);
    }

}

pub fn update_hatoverlaps(m : f64, q : f64, v : f64, alpha : f64, gamma : f64, rho : f64, delta : f64) -> (f64, f64, f64) {
    let sigma = rho - (m*m / q) + delta;

    let im = integrals::integrate_for_mhat(m, q, v, sigma,);
    let iq = integrals::integrate_for_qhat(m, q, v, sigma);
    let iv = integrals::integrate_for_vhat(m, q, v, sigma);

    let mhat = alpha * gamma.sqrt() * im / v;
    let vhat = alpha * ((1.0/v) - (1.0 / v.powi(2)) * iv);
    let qhat = alpha * iq/v.powi(2);
    
    return (mhat, qhat, vhat);

}

pub fn iterate_se(m : f64, q : f64, v : f64, alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64) -> (f64, f64, f64) {
    let (mhat, qhat, vhat) = update_hatoverlaps(m, q, v, alpha, gamma, rho, delta);
    let (new_m, new_q, new_v) =  update_overlaps(mhat, qhat, vhat, kappa1, kappastar, gamma, lambda);
    return (new_m, new_q, new_v)
}

pub fn state_evolution(alpha : f64, delta : f64, gamma : f64, kappa1 : f64, kappastar : f64, lambda : f64, rho : f64, se_tolerance : f64, relative_tolerance : bool) -> (f64, f64, f64) {
    let (mut m, mut q, mut v) = (0.01, 0.01, 0.99);
    let (mut prev_m, mut prev_q, mut prev_v) : (f64, f64, f64);
    let mut difference = 1.0;
    
    while difference > se_tolerance {
        (prev_m, prev_q, prev_v) = (m, q, v);
        (m, q, v) = iterate_se(m, q, v, alpha, delta, gamma, kappa1, kappastar, lambda, rho);
        if m == f64::NAN || q == f64::NAN || v == f64::NAN {
            panic!("One of the overlaps is NAN");
        }
        if relative_tolerance {
            difference = (m - prev_m).abs() / m.abs() + (q - prev_q).abs() / q.abs() + (v - prev_v).abs() / v.abs();
        }
        else {
            difference = (m - prev_m).abs() + (q - prev_q).abs() + (v - prev_v).abs();
        }
    }

    return (m, q, v);
}

// cd Code/EPFL/overparametrized-uncertainty/Code ; conda activate ml ; python -m experiments.amp_experiment