use crate::gcmrust::data_models::base_model;
use crate::gcmrust::utility::kappas;

pub struct GCMPrior {
    pub kappa1 : f64,
    pub kappastar : f64,
    pub gamma : f64,
    pub lambda : f64,
    pub rho : f64,
}

pub struct GCMBayesOptimalPrior {
    pub kappa1 : f64,
    pub kappastar : f64,
    pub gamma : f64,
    pub rho : f64,
}

impl base_model::ParameterPrior for GCMPrior {

    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64 {
        return self.gamma;
    }

    fn update_overlaps(&self, mhat : f64, qhat : f64, vhat : f64) -> (f64, f64, f64) {
        let alpha  = self.gamma;
        let gamma  = 1.0 / self.gamma;
            
        let sigma  = self.kappa1;
        let kk     = self.kappastar * self.kappastar;
        let alphap = ( sigma * (1.0 + alpha.sqrt())).powi(2);
        let alpham = ( sigma * (1.0 - alpha.sqrt())).powi(2);

        if self.lambda == 0.0 {
            let den    = 1.0 + kk * vhat;
            let aux    = (((alphap+kk)*vhat+1.0)*((alpham+kk) * vhat + 1.0)).sqrt();
            let aux2   = (((alphap+kk)*vhat + 1.0) / ((alpham+kk) * vhat + 1.0)).sqrt();
            let mut iv = ((kk*vhat + 1.0) * ((alphap+alpham)*vhat + 2.0) - 2.0 *kk*vhat.powi(2) * (alphap*alpham).sqrt() -2.0 * aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+1.0)*sigma.powi(2));
            iv              = iv + 0.0_f64.max(1.0 - gamma)*kk/(1.0 + vhat * kk);
            let i1     = (alphap * vhat*(-3.0 * den+aux)+4.0 * den * (-den+aux)+alpham*vhat*(-2.0 * alphap * vhat - 3.0 * den + aux))/(4.0 * alpha * vhat.powi(3) * sigma.powi(2)*aux);
            let i2     = (alphap * vhat+alpham*vhat*(1.0 - 2.0 * aux2) + 2.0 * den * (1.0 - aux2))/(4.0 * alpha * vhat.powi(2) * aux * sigma.powi(2));
            let i3     = (2.0 * vhat * alphap*alpham+(alphap+alpham) * den- 2.0 * (alphap*alpham).sqrt() * aux)/(4.0 * alpha * den.powi(2) * sigma.powi(2) * aux);
            let mut iq = (qhat + mhat.powi(2)) * i1 + (2.0*qhat+mhat.powi(2)) * kk * i2 + qhat * kk.powi(2) * i3;
            iq              = iq + 0.0_f64.max(1.0-gamma)*qhat*(kk / den).powi(2);
            let im     = ((alpham + alphap+2.0*kk)*vhat+2.0 - 2.0 * aux)/(4.0*alpha*(vhat/sigma).powi(2));
            let v = iv;
            let m = mhat * (self.gamma).sqrt() * im;
            let q = iq;
            return (m, q, v);
        }

        else {
            let den    = self.lambda+kk*vhat;
            let aux    = ( ((alphap+kk)*vhat+self.lambda) * ((alpham+kk)*vhat+self.lambda)).sqrt();
            let aux2   = ( ((alphap+kk)*vhat+self.lambda) /( (alpham+kk)*vhat+self.lambda)).sqrt();
            let mut iv = ((kk*vhat+self.lambda)*((alphap+alpham)*vhat+2.0 * self.lambda)-2.0 * kk*vhat.powi(2)*(alphap*alpham).sqrt()-2.0 * self.lambda*aux)/(4.0 * alpha*vhat.powi(2)*(kk*vhat+self.lambda)*sigma.powi(2));
            iv              = iv + f64::max(0.0, 1.0-gamma)*kk / (self.lambda + vhat * kk);
            let i1     = (alphap*vhat*(-3.0*den+aux)+4.0*den*(-den+aux)+alpham*vhat*(-2.0*alphap*vhat-3.0*den+aux))/(4.0*alpha*vhat.powi(3)*sigma.powi(2)*aux);
            let i2     = (alphap*vhat+alpham*vhat*(1.0-2.0*aux2)+2.0*den*(1.0-aux2))/(4.0*alpha*vhat.powi(2)*aux*sigma.powi(2));
            let i3     = (2.0 * vhat * alphap * alpham+(alphap + alpham) * den - 2.0 * (alphap*alpham).sqrt()* aux) / (4.0 * alpha * den.powi(2) * sigma.powi(2)* aux) ;
            let mut iq = (qhat+mhat.powi(2))*i1+(2.0*qhat+mhat.powi(2))*kk*i2+qhat*kk.powi(2)*i3;
            iq              = iq + f64::max(0.0, 1.0 - gamma)*qhat*kk.powi(2)/den.powi(2);
            let im     = ((alpham+alphap+2.0*kk)*vhat+2.0*self.lambda-2.0*aux)/(4.0*alpha*vhat.powi(2)*sigma.powi(2));


            let v = iv;
            let m = mhat * (self.gamma).sqrt() * im;
            let q = iq;
            return (m, q, v);
        }  
    }
}

impl base_model::ParameterPrior for GCMBayesOptimalPrior {
    fn get_rho(&self) -> f64 {
        return self.rho;
    }
    fn get_gamma(&self) -> f64 {
        return self.gamma;
    }
    fn update_overlaps(&self, _mhat : f64, qhat : f64, _vhat : f64) -> (f64, f64, f64) {
        let kk1 = self.kappa1 * self.kappa1;
        let kkstar = self.kappastar * self.kappastar;
        let q = self.gamma * qhat * kappas::marcenko_pastur_integral(&{|x : f64| -> f64 { (kk1 * x / (kk1 * x + kkstar)).powi(2) / (1.0 + qhat * (kk1 * x / (kk1 * x + kkstar))) }}, self.gamma);

        let m = q;
        let v = self.rho - q;
        return (m, q, v); 
    }
}