use super::{InputLayer, Layer, SigmoidLayer};
use rulinalg::{matrix::{Matrix, BaseMatrix}, vector::Vector};
use rand::Rng;

pub struct Network {
    pub layers: Vec<Box<Layer + Sync>>,
    pub w: Vec<Matrix<f64>>,
    pub b: Vec<Vector<f64>>
}

impl Network {
    pub fn new_mnist() -> Self {
        Self::new(vec![
            Box::new(InputLayer::new(784)),
            Box::new(SigmoidLayer::new(30)),
            Box::new(SigmoidLayer::new(10)),
        ])
    }

    pub fn new_test() -> Self {
        Self::new(vec![
            Box::new(InputLayer::new(5)),
            Box::new(SigmoidLayer::new(3)),
            Box::new(SigmoidLayer::new(1)),
        ])
    }

    pub fn new(layers: Vec<Box<Layer + Sync>>) -> Self {
        let mut rng = rand::thread_rng();
        let mut w : Vec<Matrix<f64>> = vec![];
        let mut b : Vec<Vector<f64>> = vec![];
        for layer in 0..layers.len() - 1 {
            let prev_n = layers[layer].output_shape();
            let cur_n = layers[layer + 1].input_shape();
            let w_mat = Matrix::new(cur_n, prev_n, 
                (0..cur_n * prev_n).map(|_| rng.gen_range(-1.0, 1.0)).collect::<Vec<f64>>());
            let b_vec = Vector::new(
                (0..cur_n).map(|_| rng.gen_range(-1.0, 1.0)).collect::<Vec<f64>>());
            w.push(w_mat);
            b.push(b_vec);
        }
        Self {
            layers, w, b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mnist_network() {
        let network = Network::new_mnist();
    }

    #[test]
    fn test_create_mnist_network_size() {
        let network = Network::new_mnist();
        assert_eq!(network.w.len(), 2);
        assert_eq!(network.b.len(), 2);
        assert_eq!(network.w[0].rows(), 30);
        assert_eq!(network.w[0].cols(), 784);
        assert_eq!(network.w[1].rows(), 10);
        assert_eq!(network.w[1].cols(), 30);
        assert_eq!(network.b[0].size(), 30);
        assert_eq!(network.b[1].size(), 10);
    }
}
