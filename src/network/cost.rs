use rulinalg::vector::Vector;

pub trait Cost {
    fn cost(&self, output: &Vector<f64>, sample: &Vector<f64>) -> f64;
    fn d_cost(&self, output: &Vector<f64>, sample: &Vector<f64>) -> Vector<f64>;
}
