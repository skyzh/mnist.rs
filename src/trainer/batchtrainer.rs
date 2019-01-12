use super::Network;
use rand::Rng;
use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    vector::Vector,
};

pub struct BatchTrainer {
    pub network: Network,
}

impl BatchTrainer {
    pub fn new() -> Self {
        Self {
            network: Network::init(&vec![784, 30, 10]),
        }
    }

    pub fn network(&self) -> &Network {
        &self.network
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
            let mut sum_network = Network::new(&vec![784, 30, 10]);
            for j in 0..batch_size {
                let mut batch_network = Network::new(&vec![784, 30, 10]);
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
                batch_network.delta(&batch, &result_vec, &self.network);
                sum_network.sum(batch_network);
            }
            self.network
                .apply(sum_network, learning_rate / batch_size as f64);
        }
    }
}
