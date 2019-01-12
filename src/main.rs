#[macro_use]
extern crate rulinalg;

mod trainer;
use self::trainer::{BatchTrainer, Dataset, Evaluator};

fn main() {
    let dataset = Dataset::new();
    let mut trainer = BatchTrainer::new();
    for epoch in 0..1000 {
        trainer.train(
            dataset.trn_size,
            &dataset.get_trn_mat(),
            &dataset.data.trn_lbl,
            20,
            1.0,
        );
        let (correct, total) = Evaluator::evaluate(&dataset, trainer.network());
        println!("Epoch {}: {}/{}", epoch, correct, total);
    }
}
