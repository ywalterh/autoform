extern crate ndarray;

use ndarray::{Array2};
use rand::prelude::*;

#[cfg(test)]
mod tests {
    use ndarray::{arr2};
    use super::*;

    #[test]
    fn test_new() {
        let x = arr2(&[[0_f32, 0_f32, 1_f32],
                         [0_f32, 1_f32, 1_f32],
                         [1_f32, 0_f32, 1_f32],
                         [1_f32, 1_f32, 1_f32]]);

        let y = arr2(&[[0_f32], [1_f32], [1_f32], [0_f32]]);
        let mut result = Network::new(x, y);

        assert_eq!(result.output.shape()[0], 4);

        // in reality, this is a 3x4 array, so len is 12. lossy way of testing it 
        assert_eq!(result.weights1.len(), 12);
        assert_ne!(result.weights1[[0, 0]], result.weights1[[0, 1]]);

        // test feed_forward_once
        result.feed_forward();

        // see if feedforward is working
        assert_ne!(result.layer1[[0, 0]], 0_f32);

        let before_propagate = result.weights2[[0,0]];
        result.back_propagation();
        assert_ne!(result.weights2[[0,0]], before_propagate);
    }

    #[test]
    fn test_training() {
        //0, 0, 1 -> 0
        //0, 1, 1 -> 1,
        //1, 0, 1 -> 1,
        //1, 1, 1 -> 0

        let x = arr2(&[[0_f32, 0_f32, 1_f32],
                         [0_f32, 1_f32, 1_f32],
                         [1_f32, 0_f32, 1_f32],
                         [1_f32, 1_f32, 1_f32]]);

        let y = arr2(&[[0_f32], [1_f32], [1_f32], [0_f32]]);

        let mut network = Network::new(x, y);

        for _ in 1..2500 {
            network.feed_forward();
            network.back_propagation(); 
        }
        
        println!("Actual:\n {}", network.y);
        println!("Predication:\n {}", network.output);

        // calculate means to make sure it's smaller than 0.001
        let cost_square = (network.y - network.output).mapv(square);
        let mean = cost_square.sum() / cost_square.len() as f32;
        
        println!("Loss is :\n {}", mean);
        assert!(mean < 0.001_f32);
    }

    fn square(x: f32) -> f32 {
        x * x
    }
}

// In the example, input is 3x1
// output is a sinlge value
#[derive(Debug, Default)]
pub struct Network {
    // input is 1x3 matrix
    input: Array2<f32>,
    //3x4 matrix
    weights1: Array2<f32>,
    //4x1 matrix
    weights2: Array2<f32>,

    // since all these are dimensional computation
    // both y and f32 are 1x1 matrix
    y: Array2<f32>,
    output: Array2<f32>,

    // layer1: 1x4 = sigmoid(input * weights1)
    // 1x3 dot_prod 3x4
    layer1: Array2<f32>,
}

// Initialize the NN with empty layers
// and input outpu as arrrys?
// Having some troubele translating things along
impl Network {
    fn new(x: Array2<f32>, y: Array2<f32>) -> Network {
        let mut rng = rand::thread_rng();
        // use the shape of x to determine what the shape of weights1 has
        // should be safe to retrieve shape[1] here because x is
        // of 2D array
        let mut weights1 = Array2::<f32>::zeros((x.shape()[1], 4));
        let mut weights2 = Array2::<f32>::zeros((4, 1));
        // iterate weights to initialize it with random numbers
        for mut row in weights1.genrows_mut() {
            for i in 0..row.len() {
                row[i] = rng.gen_range(0_f32, 1_f32);
            }
        }
        for mut row in weights2.genrows_mut() {
            row[0] = rng.gen_range(0_f32, 1_f32);
        }

        // explicitly make a copy of the value of len of y
        // not usd yet, but let's change it to multi-dimensional later
        // to_vec implemennts Clone trait, lol
        let y_shape = y.shape().to_vec();
        Network {
            input: x,
            weights1,
            weights2,
            y,
            output: Array2::zeros((y_shape[0], y_shape[1])),

            // do we need to pre-assign values?
            // guess it doesn't matter because we'll use
            // them later anyway
            layer1: Array2::zeros((1, 4)),
        }
    }

    fn feed_forward(&mut self) {
        // add the result each iteration  to output
        self.layer1 = self.input.dot(&self.weights1).mapv_into(Network::sigmoid);
        self.output = self.layer1.dot(&self.weights2).mapv_into(Network::sigmoid);
    }

    fn back_propagation(&mut self) {
        // application of the chain rule to find derivative of 
        // the loss function with respect to weights2 and weights1
        let z = 2_f32 * (&self.y - &self.output) * &self.output.mapv(Network::sigmoid_derivative);
        let d_weights2 = self.layer1.t().dot(&z);
        let d_weights1 = self.input.t().dot(&(z.dot(&self.weights2.t()) * &self.layer1.mapv(Network::sigmoid_derivative)));
       
        self.weights1 = &self.weights1 + &d_weights1;
        self.weights2 = &self.weights2 + &d_weights2;
    }


    fn sigmoid(x: f32) -> f32 {
        1_f32 / (1_f32 + (-x).exp())
    }

    fn sigmoid_derivative(x: f32) -> f32 {
        x * (1_f32 - x)
    }
}
