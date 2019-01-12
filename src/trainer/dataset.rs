use mnist::{Mnist, MnistBuilder};

pub struct Dataset {
    pub data: Mnist,
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
            placeholder: true,
        }
    }
}
