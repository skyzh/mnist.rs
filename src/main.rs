#[macro_use]
extern crate rulinalg;

use rand::prelude::*;
use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    vector::Vector,
};

mod network;
use self::network::{Network, MSE};
mod dataset;
use self::dataset::Dataset;
mod trainer;
use self::trainer::{Trainer, SGD};

fn main() {
    let network = Network::new_mnist();
    let dataset = Dataset::new();
    let mut sgd = SGD::new(network);
    let mut rng = rand::thread_rng();
    let mini_batch = 10;
    let cost = MSE {};
    let targets: Vec<Vector<f64>> = (0..10)
        .map(|i| Vector::from_fn(10, |idx| if i == idx { 1.0 } else { 0.0 }))
        .collect();
    for epoch in 0..100 {
        let mut train_dataset: Vec<u32> = (1..dataset.trn_size).collect();
        train_dataset.shuffle(&mut rng);
        for i in 0..(dataset.trn_size / mini_batch) as usize {
            let mut nablas = vec![];
            for _ in 0..mini_batch as usize {
                let input = &dataset.trn_img[i];
                let (xs, activations) = sgd.feed_forward(input);
                let nabla = sgd.back_prop(
                    &cost,
                    &xs,
                    &activations,
                    &targets[dataset.trn_lbl[i] as usize],
                );
                nablas.push(nabla);
            }
            for j in 0..mini_batch as usize {
                sgd.apply(&nablas[j], 1.0 / mini_batch as f64);
            }
        }
        let mut correct = 0;
        for i in 0..dataset.tst_size {
            let input = &dataset.tst_img[i as usize];
            let (_, activations) = sgd.feed_forward(input);
            let activation = activations.last().unwrap().argmax();
            if activation.0 == dataset.tst_lbl[i as usize] as usize {
                correct += 1;
            }
        }
        println!("Epoch {}, {}/{}", epoch, correct, dataset.tst_size);
    }
}
