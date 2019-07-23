use rulinalg::{matrix::Matrix, vector::Vector};

use super::Layer;

pub struct SigmoidLayer {
    neurons: usize
}

impl SigmoidLayer {
    fn new(neurons: usize) -> Self {
        return Self {
            neurons
        }
    }
}

impl Layer for SigmoidLayer {
    fn output_shape(&self) -> usize { self.neurons }
    fn input_shape(&self) -> usize { self.neurons }
    fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64> {
        debug_assert!(input.size() == self.input_shape());
        return vector![1.0];
    }
    fn back_prop(&self, nabla: &Vector<f64>) -> Vector<f64> {
        debug_assert!(nabla.size() == self.output_shape());
        return vector![1.0];
    }
}
