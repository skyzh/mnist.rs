#[macro_use]
extern crate rulinalg;

mod trainer;
use self::trainer::{Dataset, Evaluator, Network};

fn main() {
    let dataset = Dataset::new();
    let mut network = Network::init(&vec![784, 30, 10]);
    for epoch in 0..1000 {
        network.train(
            dataset.trn_size,
            &dataset.get_trn_mat(),
            &dataset.data.trn_lbl,
            20,
            3.0,
        );
        let (correct, total) = Evaluator::evaluate(&dataset, &network);
        println!("Epoch {}: {}/{}", epoch, correct, total);
    }
}
