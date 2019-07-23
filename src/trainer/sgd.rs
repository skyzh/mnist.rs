use rulinalg::{matrix::Matrix, vector::Vector};

use super::Trainer;
use crate::network::Cost;
use crate::network::Network;

pub struct SGD {
    network: Network,
}

impl SGD {
    pub fn new(network: Network) -> Self {
        SGD { network }
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
            let input = &self.network.w[i] * output + &self.network.b[i];
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
        w: &Vec<Matrix<f64>>,
        b: &Vec<Vector<f64>>,
    ) -> (Vec<Matrix<f64>>, Vec<Vector<f64>>) {
        return (vec![], vec![]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd() {
        let network = Network::new_test();
        let sgd = SGD::new(network);
        let (activations, xs) = sgd.feed_forward(&vector![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
