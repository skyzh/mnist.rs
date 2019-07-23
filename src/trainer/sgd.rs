use rand::Rng;
use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    vector::Vector,
};

use super::Trainer;
use crate::network::Cost;
use crate::network::Network;

pub struct SGD {
    network: Network,
    w: Vec<Matrix<f64>>,
    b: Vec<Vector<f64>>,
}

impl SGD {
    pub fn new(network: Network) -> Self {
        let mut rng = rand::thread_rng();
        let mut w: Vec<Matrix<f64>> = vec![];
        let mut b: Vec<Vector<f64>> = vec![];
        let layers = &network.layers;
        for layer in 0..layers.len() - 1 {
            let prev_n = layers[layer].output_shape();
            let cur_n = layers[layer + 1].input_shape();
            let w_mat = Matrix::new(
                cur_n,
                prev_n,
                (0..cur_n * prev_n)
                    .map(|_| rng.gen_range(-1.0, 1.0))
                    .collect::<Vec<f64>>(),
            );
            let b_vec = Vector::new(
                (0..cur_n)
                    .map(|_| rng.gen_range(-1.0, 1.0))
                    .collect::<Vec<f64>>(),
            );
            w.push(w_mat);
            b.push(b_vec);
        }
        Self { network, w, b }
    }
}

impl Trainer for SGD {
    fn feed_forward(
        &self,
        first_layer_output: &Vector<f64>,
    ) -> (Vec<Vector<f64>>, Vec<Vector<f64>>) {
        let mut activations: Vec<Vector<f64>> = vec![];
        let mut xs: Vec<Vector<f64>> = vec![];
        let mut output = &first_layer_output.clone();
        for i in 0..self.network.layers.len() - 1 {
            let input = &self.w[i] * output + &self.b[i];
            let activation = self.network.layers[i + 1].feed_forward(&input);
            activations.push(activation);
            xs.push(input);
            output = activations.last().unwrap();
        }
        (activations, xs)
    }

    fn back_prop(
        &self,
        cost: &impl Cost,
        xs: Vec<Vector<f64>>,
        activations: Vec<Vector<f64>>,
        target: &Vector<f64>,
    ) -> (Vec<Matrix<f64>>, Vec<Vector<f64>>) {
        let mut nabla = cost.d_cost(activations.last().unwrap(), target);
        let mut nabla_w: Vec<Matrix<f64>> = vec![];
        let mut nabla_b: Vec<Vector<f64>> = vec![];

        for i in (0..self.network.layers.len() - 1).rev() {
            let layer = &self.network.layers[i + 1];
            let delta = layer.back_prop(&nabla, &xs[i]);
            nabla = self.w[i].transpose() * &delta;
            nabla_w.push(
                Matrix::from(delta.clone()) * Matrix::from(activations[i].clone()).transpose(),
            );
            nabla_b.push(delta);
        }
        nabla_w.reverse();
        nabla_b.reverse();
        return (nabla_w, nabla_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let network = Network::new_test();
        let sgd = SGD::new(network);
        let (activations, xs) = sgd.feed_forward(&vector![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(activations.len(), 2);
        assert_eq!(xs.len(), 2);
        assert_eq!(activations[0].size(), 3);
        assert_eq!(activations[1].size(), 1);
        assert_eq!(xs[0].size(), 3);
        assert_eq!(xs[1].size(), 1);
    }

    #[test]
    fn test_back_prop() {
        let network = Network::new_test();
        let sgd = SGD::new(network);
        let (activations, xs) = sgd.feed_forward(&vector![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(activations.len(), 2);
        assert_eq!(xs.len(), 2);
        assert_eq!(activations[0].size(), 3);
        assert_eq!(activations[1].size(), 1);
        assert_eq!(xs[0].size(), 3);
        assert_eq!(xs[1].size(), 1);
    }

    #[test]
    fn test_create_mnist_network_size() {
        let sgd = SGD::new(Network::new_mnist());
        assert_eq!(sgd.w.len(), 2);
        assert_eq!(sgd.b.len(), 2);
        assert_eq!(sgd.w[0].rows(), 30);
        assert_eq!(sgd.w[0].cols(), 784);
        assert_eq!(sgd.w[1].rows(), 10);
        assert_eq!(sgd.w[1].cols(), 30);
        assert_eq!(sgd.b[0].size(), 30);
        assert_eq!(sgd.b[1].size(), 10);
    }
}
