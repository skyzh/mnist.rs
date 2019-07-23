use rulinalg::vector::Vector;

use super::Layer;

pub struct SigmoidLayer {
    neurons: usize
}

impl SigmoidLayer {
    pub fn new(neurons: usize) -> Self {
        return Self {
            neurons
        }
    }

    fn sigmoid(x: f64) -> f64 {
        return 1.0 / (1.0 + (-x).exp())
    }

    fn d_sigmoid(x: f64) -> f64 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }
}

impl Layer for SigmoidLayer {
    fn output_shape(&self) -> usize { self.neurons }
    fn input_shape(&self) -> usize { self.neurons }
    fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64> {
        debug_assert!(input.size() == self.input_shape());
        input.clone().apply(&|x| Self::sigmoid(x))
    }
    fn back_prop(&self, nabla: &Vector<f64>, x: &Vector<f64>) -> Vector<f64> {
        debug_assert!(nabla.size() == self.output_shape());
        nabla.elemul(&x.clone().apply(&|x| Self::d_sigmoid(x)))
    }
}
