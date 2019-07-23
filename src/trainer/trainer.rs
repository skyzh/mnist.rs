use rulinalg::{matrix::Matrix, vector::Vector};

use crate::network::Cost;

pub trait Trainer {
    fn feed_forward(&self, input: &Vector<f64>) -> (Vec<Vector<f64>>, Vec<Vector<f64>>);
    fn back_prop(
        &self,
        cost: &impl Cost,
        xs: Vec<Vector<f64>>,
        activations: Vec<Vector<f64>>,
        target: &Vector<f64>,
        w: &Vec<Matrix<f64>>,
        b: &Vec<Vector<f64>>,
    ) -> (Vec<Matrix<f64>>, Vec<Vector<f64>>);
}
