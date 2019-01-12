pub struct Evaluator {}
use super::{Dataset, Network};
use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    vector::Vector,
};

impl Evaluator {
    pub fn evaluate(dataset: &Dataset, network: &Network) -> (u32, u32) {
        let mut cnt = 0;
        let tst_image = dataset.get_tst_mat();
        for i in 0..10000 {
            let row_begin = i * 28;
            let batch = tst_image
                .select_rows(&(row_begin as usize..(row_begin + 28) as usize).collect::<Vec<_>>());
            let batch = Vector::new(batch.iter().map(|v| *v).collect::<Vec<f64>>());
            let (idx, val) = network.feedforward(&batch).argmax();
            if idx as u8 == dataset.data.tst_lbl[i] {
                cnt = cnt + 1
            }
        }
        (cnt, 10000)
    }
}
