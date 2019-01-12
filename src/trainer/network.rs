use rand::Rng;
use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    vector::Vector,
};
use std::iter::{Map, Zip};

pub struct Network {
    layers: Vec<usize>,
    biases: Vec<Vector<f64>>,
    weights: Vec<Matrix<f64>>,
}

impl Network {
    pub fn new(layers: &Vec<usize>) -> Self {
        Self {
            layers: layers.clone(),
            biases: (1..layers.len())
                .map(|idx| Vector::<f64>::zeros(layers[idx]))
                .collect(),
            weights: (0..(layers.len() - 1))
                .map(|idx| Matrix::<f64>::zeros(layers[idx + 1], layers[idx]))
                .collect(),
        }
    }

    pub fn init(layers: &Vec<usize>) -> Self {
        Self {
            layers: layers.clone(),
            biases: (1..layers.len())
                .map(|idx| Vector::<f64>::from_fn(layers[idx], |_| rand::thread_rng().gen::<f64>()))
                .collect(),
            weights: (0..(layers.len() - 1))
                .map(|idx| {
                    Matrix::<f64>::from_fn(layers[idx + 1], layers[idx], |_, _| {
                        rand::thread_rng().gen::<f64>()
                    })
                })
                .collect(),
        }
    }

    pub fn feedforward(&self, input: &Vector<f64>) -> Vector<f64> {
        let mut output: Vector<f64> = input.clone();
        for (bias, weight) in self.biases.iter().zip(self.weights.iter()) {
            output = (weight * &output + bias).apply(&|x| Self::sigmoid(x));
        }
        output
    }

    pub fn delta(&mut self, input: &Vector<f64>, output: &Vector<f64>, network: &Network) {
        let mut activations = Vec::<Vector<f64>>::new();
        let mut activation_xs = Vec::<Vector<f64>>::new();
        activations.push(input.clone());

        for idx in 0..self.biases.len() {
            let activation = &network.weights[idx] * &activations[idx] + &network.biases[idx];
            activations.push(activation.clone().apply(&|x| Self::sigmoid(x)));
            activation_xs.push(activation);
        }

        let mut delta = Self::d_cost((&activations).last().unwrap(), output).elemul(
            &activation_xs
                .last()
                .unwrap()
                .clone()
                .apply(&|x| Self::d_sigmoid(x)),
        );

        let second_to_last_activation = activations[activations.len() - 2].clone();
        let second_to_last_activation = Self::transpose(second_to_last_activation);
        *self.weights.last_mut().unwrap() = Self::to_mat(delta.clone()) * second_to_last_activation;
        *self.biases.last_mut().unwrap() = delta.clone();

        for idx in (0..self.biases.len() - 1).rev() {
            let sp = (&activation_xs[idx]).clone().apply(&|x| Self::d_sigmoid(x));
            delta = (network.weights[idx + 1].transpose() * delta).elemul(&sp);
            self.weights[idx] =
                Self::to_mat(delta.clone()) * Self::transpose(activations[idx].clone());
            self.biases[idx] = delta.clone();
        }
    }

    pub fn apply(&mut self, network: Network, learning_rate: f64) {
        for (bias, delta_bias) in self.biases.iter_mut().zip(network.biases.iter()) {
            *bias += delta_bias * (-learning_rate);
        }
        for (weight, delta_weight) in self.weights.iter_mut().zip(network.weights.iter()) {
            *weight += delta_weight * (-learning_rate);
        }
    }

    pub fn sum(&mut self, network: Network) {
        for (bias, delta_bias) in self.biases.iter_mut().zip(network.biases.iter()) {
            *bias += delta_bias;
        }
        for (weight, delta_weight) in self.weights.iter_mut().zip(network.weights.iter()) {
            *weight += delta_weight;
        }
    }

    pub fn debug(&self) {
        println!("{}", self.biases[1]);
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn d_sigmoid(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    fn d_cost(output: &Vector<f64>, target: &Vector<f64>) -> Vector<f64> {
        output - target
    }

    fn to_mat(vec: Vector<f64>) -> Matrix<f64> {
        Matrix::new(vec.size(), 1, vec)
    }

    fn transpose(vec: Vector<f64>) -> Matrix<f64> {
        Self::to_mat(vec).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::{BaseMatrix, Matrix, Network, Vector};

    #[test]
    fn test_new() {
        let network = Network::new(&vec![800 as usize, 3 as usize, 10 as usize]);
        assert_eq!(network.layers.len(), 3);
        assert_eq!(network.biases.len(), 2);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.weights[0].rows(), 3);
        assert_eq!(network.weights[0].cols(), 800);
        assert_eq!(network.weights[1].rows(), 10);
        assert_eq!(network.weights[1].cols(), 3);
        assert_eq!(network.biases[0].size(), 3);
        assert_eq!(network.biases[1].size(), 10);
    }

    #[test]
    fn test_init() {
        let network = Network::init(&vec![800 as usize, 3 as usize, 10 as usize]);
    }

    #[test]
    fn test_feedforward() {
        let network = Network::new(&vec![
            3 as usize, 5 as usize, 6 as usize, 7 as usize, 4 as usize,
        ]);
        assert_eq!(
            network.feedforward(&Vector::new(vec![0.0, 0.0, 0.0])),
            Vector::new(vec![0.5, 0.5, 0.5, 0.5])
        );
    }

    #[test]
    fn test_train() {
        let mut network = Network::new(&vec![1 as usize, 10 as usize, 4 as usize]);
        let _network = Network::init(&vec![1 as usize, 10 as usize, 4 as usize]);
        println!(
            "Layer 0 -> Layer 1\n{}\n{}",
            network.weights[0], network.biases[0]
        );
        println!(
            "Layer 1 -> Layer 2\n{}\n{}",
            network.weights[1], network.biases[1]
        );
        network.delta(&vector![1.0], &vector![1.0, 2.0, 3.0, 4.0], &_network);
        println!(
            "Layer 0 -> Layer 1\n{}\n{}",
            network.weights[0], network.biases[0]
        );
        println!(
            "Layer 1 -> Layer 2\n{}\n{}",
            network.weights[1], network.biases[1]
        );
        network.delta(&vector![-1.0], &vector![-1.0, -2.0, -3.0, -4.0], &_network);
        println!(
            "Layer 0 -> Layer 1\n{}\n{}",
            network.weights[0], network.biases[0]
        );
        println!(
            "Layer 1 -> Layer 2\n{}\n{}",
            network.weights[1], network.biases[1]
        );
    }
}
