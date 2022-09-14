pub trait Channel {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64;
    fn df0(&self, y : f64, omega : f64, v : f64) -> f64;
}