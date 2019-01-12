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
                .map(|idx| Vector::<f64>::from_fn(layers[idx], |_| rand::thread_rng().gen_range(-1.0 as f64, 1.0 as f64)))
                .collect(),
            weights: (0..(layers.len() - 1))
                .map(|idx| {
                    Matrix::<f64>::from_fn(layers[idx + 1], layers[idx], |_, _| {
                        rand::thread_rng().gen_range(-1.0 as f64, 1.0 as f64)
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

    pub fn backprop(&mut self, input: &Vector<f64>, output: &Vector<f64>) -> Self {
        let mut delta_network = Self::new(&self.layers);
        let mut activations = Vec::<Vector<f64>>::new();
        let mut activation_xs = Vec::<Vector<f64>>::new();
        activations.push(input.clone());

        for idx in 0..self.biases.len() {
            let activation = &self.weights[idx] * &activations[idx] + &self.biases[idx];
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
        *delta_network.weights.last_mut().unwrap() =
            Self::to_mat(delta.clone()) * second_to_last_activation;
        *delta_network.biases.last_mut().unwrap() = delta.clone();

        for idx in (0..self.biases.len() - 1).rev() {
            let sp = (&activation_xs[idx]).clone().apply(&|x| Self::d_sigmoid(x));
            delta = (self.weights[idx + 1].transpose() * delta).elemul(&sp);
            delta_network.weights[idx] =
                Self::to_mat(delta.clone()) * Self::transpose(activations[idx].clone());
            delta_network.biases[idx] = delta.clone();
        }
        delta_network
    }
    pub fn train(
        &mut self,
        trn_size: u32,
        trn_image: &Matrix<f64>,
        trn_label: &Vec<u8>,
        batch_size: u32,
        learning_rate: f64,
    ) {
        let mut shuffle_batch: Vec<u32> = (0..trn_size).collect();
        rand::thread_rng().shuffle(&mut shuffle_batch);

        for i in 0..trn_size / batch_size {
            let mut nabla_network = Self::new(&self.layers);
            for j in 0..batch_size {
                let batch_id = shuffle_batch[(i * batch_size + j) as usize];
                let row_begin = batch_id * 28;
                let batch = trn_image.select_rows(
                    &(row_begin as usize..(row_begin + 28) as usize).collect::<Vec<_>>(),
                );
                let batch = Vector::new(batch.iter().map(|v| *v).collect::<Vec<f64>>());
                let result_vec = Vector::from_fn(10, |idx| {
                    if trn_label[batch_id as usize] as u32 == idx as u32 {
                        1.0
                    } else {
                        0.0
                    }
                });
                let delta_network = self.backprop(&batch, &result_vec);
                nabla_network.apply(delta_network, 1.0);
            }
            self.apply(nabla_network, -learning_rate / batch_size as f64);
        }
    }

    fn apply(&mut self, network: Network, multiplexer: f64) {
        for i in 0..self.biases.len() {
            self.biases[i] += &network.biases[i] * multiplexer;
        }
        for i in 0..self.weights.len() {
            self.weights[i] += &network.weights[i] * multiplexer;
        }
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
}
