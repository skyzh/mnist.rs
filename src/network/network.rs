use super::{InputLayer, Layer, SigmoidLayer};

pub struct Network {
    pub layers: Vec<Box<Layer + Sync>>
}

impl Network {
    pub fn new_mnist() -> Self {
        Self::new(vec![
            Box::new(InputLayer::new(784)),
            Box::new(SigmoidLayer::new(30)),
            Box::new(SigmoidLayer::new(10)),
        ])
    }

    pub fn new_test() -> Self {
        Self::new(vec![
            Box::new(InputLayer::new(5)),
            Box::new(SigmoidLayer::new(3)),
            Box::new(SigmoidLayer::new(1)),
        ])
    }

    pub fn new(layers: Vec<Box<Layer + Sync>>) -> Self {
        Self { layers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_mnist_network() {
        let _ = Network::new_mnist();
    }
}
