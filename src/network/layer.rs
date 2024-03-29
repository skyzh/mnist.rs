use rulinalg::vector::Vector;

pub trait Layer {
    fn output_shape(&self) -> usize;
    fn input_shape(&self) -> usize;
    fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64>;
    fn back_prop(&self, nabla: &Vector<f64>, x: &Vector<f64>) -> Vector<f64>;
}
