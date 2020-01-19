use ndarray::{Array, Array1, Array2};
use rand::prelude::*;
use fastapprox::fast::sigmoid;

// In the example, input is a array of 3
// output is a sinlge value
#[derive(Debug, Default)]
pub struct Network {
    input: Array1<f32>,
    weights1: Array2<f32>,
    weights2: Array2<f32>,
    //@Next step, makes this work on multi-dimensional
    y: f32,
    output: f32,
    layer1: Array1<f32>,
    layer2: Array1<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let x = Array1::<f32>::zeros(3);
        let mut result = Network::new(x, 0_f32);

        assert_eq!(result.output, 0_f32);

        // in reality, this is a 3x4 array, so len is 12
        assert_eq!(result.weights1.len(), 12);

        // test feed_forward_once
        result.feed_forward();
        assert_ne!(result.layer1[0], 0_f32);
        assert_ne!(result.layer2[0], 0_f32);
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_ne!(Network::sigmoid_derivative(1_f32), 0_f32);
    }
}

// Initialize the NN with empty layers
// and input outpu as arrrys?
// Having some troubele translating things along
impl Network {
    fn new(x: Array1<f32>, y: f32) -> Network {
        let mut rng = rand::thread_rng();
        let mut weights1 = Array::zeros((x.len(), 4));
        let mut weights2 = Array::zeros((4, 1));

        // iterate weights to initialize it with random numbers
        for mut row in weights1.genrows_mut() {
            row.fill(rng.gen_range(0_f32, 1_f32));
        }
        for mut row in weights2.genrows_mut() {
            row.fill(rng.gen_range(0_f32, 1_f32));
        }

        // explicitly make a copy of the value of len of y
        // not usd yet, but let's change it to multi-dimensional later
        //let y_shape = y.len();
        Network {
            input: x,
            weights1,
            weights2,
            y,
            output: 0_f32, 
            // do we need to pre-assign values?
            // guess it doesn't matter because we'll use 
            // them later anyway
            layer1: Array1::zeros(4),
            layer2: Array1::zeros(4),
        }
    }

    fn feed_forward(&mut self) {
        self.layer1 = self.input.dot(&self.weights1);
        for mut row in self.layer1.genrows_mut() {
            row.fill(sigmoid(row[0]));
        }

        self.layer2 = self.layer1.dot(&self.weights2);
        for mut row in self.layer1.genrows_mut() {
            row.fill(sigmoid(row[0]));
        }
    }

    fn back_propagation(&mut self) {
        // application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        //let d_weights2 = self.layer1.dot(2_f32 * (self.y - self.output) * sigmoid_derivative(self.output));
        //let d_weights1 = self.input.dot()


    }

    fn sigmoid_derivative(x: f32) -> f32 {
       return sigmoid(x) * (1_f32 - sigmoid(x));
    }
}
