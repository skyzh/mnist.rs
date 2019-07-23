use rulinalg::vector::Vector;

use super::Layer;

pub struct InputLayer {
    neurons: usize
}

impl InputLayer {
    pub fn new(neurons: usize) -> Self {
        InputLayer {
            neurons
        }
    }
}

impl Layer for InputLayer {
    fn output_shape(&self) -> usize { self.neurons }
    fn input_shape(&self) -> usize { self.neurons }
    fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64> {
        debug_assert!(input.size() == self.input_shape());
        input.clone()
    }
    fn back_prop(&self, nabla: &Vector<f64>, _x: &Vector<f64>) -> Vector<f64> {
        debug_assert!(nabla.size() == self.output_shape());
        panic!("cannot back prop on input layer!")
    }
}
