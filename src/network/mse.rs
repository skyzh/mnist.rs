use rulinalg::vector::Vector;
use super::Cost;

pub struct MSE {}

impl Cost for MSE {
    fn cost(&self, output: &Vector<f64>, sample: &Vector<f64>) -> f64 {
        let delta = output - sample;
        return 0.5 * (&delta.elemul(&delta)).sum();
    }
    fn d_cost(&self, output: &Vector<f64>, sample: &Vector<f64>) -> Vector<f64> {
        return output - sample
    }
}

#[cfg(test)]
mod tests {
    use super::{MSE, Cost};

    #[test]
    fn test_dcost() {
        let output = vector![1.0, 2.0, 3.0];
        let sample = vector![1.0, -2.0, 3.0];
        let cost = MSE {};
        assert_eq!(cost.d_cost(&output, &sample), vector![0.0, 4.0, 0.0]);
    }

    #[test]
    fn test_cost() {
        let output = vector![1.0, 2.0, 3.0];
        let sample = vector![1.0, -2.0, 3.0];
        let cost = MSE {};
        assert_eq!(cost.cost(&output, &sample), 8.0);
    }
}
