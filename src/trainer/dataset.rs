use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, Matrix};

pub struct Dataset {
    pub data: Mnist,
    pub trn_size: u32,
    pub rows: u32,
    pub cols: u32,
    placeholder: bool,
}

impl Dataset {
    pub fn new() -> Self {
        let (trn_size, rows, cols) = (50_000, 28, 28);
        // Deconstruct the returned Mnist struct.
        Self {
            data: MnistBuilder::new()
                .label_format_digit()
                .training_set_length(trn_size)
                .validation_set_length(10_000)
                .test_set_length(10_000)
                .finalize(),
            trn_size,
            rows,
            cols,
            placeholder: true,
        }
    }

    fn get_mat(&self, src: Vec<u8>, size: u32) -> Matrix<f64> {
        Matrix::new((size * self.rows) as usize, self.cols as usize, src)
            .try_into()
            .unwrap()
            / 255.0
    }

    pub fn get_trn_mat(&self) -> Matrix<f64> {
        self.get_mat(self.data.trn_img.clone(), self.trn_size)
    }

    pub fn get_tst_mat(&self) -> Matrix<f64> {
        self.get_mat(self.data.tst_img.clone(), 10000)
    }
}
