use mnist::{Mnist, MnistBuilder};
use rulinalg::vector::Vector;

pub struct Dataset {
    data: Mnist,
    pub trn_img: Vec<Vector<f64>>,
    pub tst_img: Vec<Vector<f64>>,
    pub trn_lbl: Vec<u8>,
    pub tst_lbl: Vec<u8>,
    pub trn_size: u32,
    pub tst_size: u32,
    pub rows: usize,
    pub cols: usize,
    pub pixels: usize,
}

impl Dataset {
    pub fn new() -> Self {
        let (trn_size, tst_size, rows, cols) = (50_000, 10_000, 28, 28);
        let data = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(trn_size)
            .validation_set_length(tst_size)
            .test_set_length(tst_size)
            .finalize();
        let pixels = rows * cols;
        let trn_img: Vec<f64> = data.trn_img.iter().map(|x| *x as f64 / 255.0).collect();
        let tst_img: Vec<f64> = data.tst_img.iter().map(|x| *x as f64 / 255.0).collect();
        let trn_img: Vec<Vector<f64>> = trn_img.chunks(pixels).map(&|v| Vector::new(v)).collect();
        let tst_img: Vec<Vector<f64>> = tst_img.chunks(pixels).map(&|v| Vector::new(v)).collect();
        let trn_lbl = data.trn_lbl.clone();
        let tst_lbl = data.tst_lbl.clone();
        Self {
            data,
            trn_img,
            tst_img,
            trn_lbl,
            tst_lbl,
            trn_size,
            tst_size,
            rows,
            cols,
            pixels,
        }
    }
}
