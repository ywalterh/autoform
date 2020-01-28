extern crate ndarray;

use ndarray::{arr2, Array2};
use rand::prelude::*;

#[cfg(test)]
mod tests {
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

        // in reality, this is a 3x4 array, so len is 4
        assert_eq!(result.weights1.len(), 12);
        assert_ne!(result.weights1[[0, 0]], result.weights1[[0, 1]]);

        // test feed_forward_once
        result.feed_forward();

        // see if feedforward is working
        assert_ne!(result.layer1[[0, 0]], 0_f32);

        let before_propagate = result.weights2[[0,0]];
        result.back_propagation();
        assert_ne!(result.weights2[[0,0]], before_propagate);

        dbg!(&result);
        assert_ne!(0, 0);
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_ne!(Network::sigmoid_derivative(1_f32), 0_f32);
    }

    #[test]
    fn test_array2_sigmoid_derivative() {
        let input_x = Array2::ones((1, 1));
        let result = Network::array2_sigmoid_derivative(&input_x);
        assert_eq!(result.len(), 1);
        assert_ne!(result[[0, 0]], 1_f32);
    }

    #[test]
    fn test_get_2d_array_of_matrix() {
        let x = arr2(&[[0_f32, 0_f32, 1_f32],
                     [0_f32, 1_f32, 1_f32],
                     [1_f32, 0_f32, 1_f32],
                     [1_f32, 1_f32, 1_f32]]);

        assert_eq!(Network::get_2d_array_of_matrix(&x, 1), arr2(&[[0_f32, 1_f32, 1_f32]]));
    }
    
    #[test]
    fn test_single() {
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

        for _ in 1..5 {
            network.feed_forward();
            network.back_propagation();
        }
        
        println!("Actual:\n {}", network.y);
        println!("Predication:\n {}", network.output);
        assert_eq!(0, 1);
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

        for _ in 1..50 {
            network.feed_forward();
            network.back_propagation(); 
        }
        
        println!("Actual:\n {}", network.y);
        println!("Predication:\n {}", network.output);
        assert_eq!(0, 1);
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
        let mut cur = 0;
        for mut output_row in self.output.genrows_mut() {
            let mut row_2d = Array2::zeros((1, 3));
            for mut row1 in row_2d.genrows_mut() {
                for i in 0..row1.len() {
                    //@Cleanup remove unwrap
                    row1[i] = *self.input.get((cur, i)).unwrap();
                }
            }

            self.layer1 = row_2d.dot(&self.weights1);
            for mut row in self.layer1.genrows_mut() {
                for i in 0..row.len() {
                    row[i] = Network::sigmoid(row[i]);
                }
            }

            // fill up the resuling 1d array
            let mut one_d_array = self.layer1.dot(&self.weights2);
            for mut row in one_d_array.genrows_mut() {
                row.fill(Network::sigmoid(row[0]));
            }

            //@Cleanup remoev unwrap
            output_row.fill(*one_d_array.get((0, 0)).unwrap());

            // increase index
            cur = cur + 1;
        }

        dbg!(&self.output);
    }

    fn get_2d_array_of_matrix(x: &Array2<f32>,  index: usize) -> Array2<f32> {
        let mut row_2d = Array2::<f32>::zeros((1, x.shape()[1]));
        
        for mut row in row_2d.genrows_mut() {
            for i in 0..row.len() {
                //@Cleanup unwrap
                row[i] = *x.get((index, i)).unwrap();
            }
        }
        
        row_2d
    }

    fn back_propagation(&mut self) {
        // application of the chain rule to find derivative of 
        // the loss function with respect to weights2 and weights1
        let mut d_weights1_total = Array2::<f32>::zeros((3, 4));
        let mut d_weights2_total = Array2::<f32>::zeros((4, 1));

        let mut cur = 0;
        for row in self.output.genrows() {
            //@Cleanup unwrap
            let cost_derivative = *self.y.get((cur, 0)).unwrap() - row[0];
            dbg!(&cost_derivative);
            let z = 2_f32 * cost_derivative * Network::array2_sigmoid_derivative(&Network::get_2d_array_of_matrix(&self.output, cur));
            let d_weights2 = self.layer1.t().dot(&z);

            let d_weights1 = &(Network::get_2d_array_of_matrix(&self.input, cur).t()).dot(&(z.dot(&self.weights2.t()) * Network::array2_sigmoid_derivative(&self.layer1)));
           
            d_weights1_total = d_weights1_total +  d_weights1;
            d_weights2_total = d_weights2_total +  d_weights2;
            cur = cur + 1;
        }

        self.weights1 = &self.weights1 + &(1_f32 * &d_weights1_total);
        self.weights2 = &self.weights2 + &(1_f32 * &d_weights2_total);
    }

    // borrow the value to perform sigmoid derivative to one
    // one dimensional array and return the resulting array
    fn array2_sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros((x.shape()[0], x.shape()[1]));

        // iterate over row to add data
        let mut index = 0;
        for mut row in result.genrows_mut() {
            // this should be correct since it's already filled,
            // the recommended approach is to use .fill
            // which obviously doesn't work for us here
            for i in 0..row.len() {
                row[i] = Network::sigmoid_derivative(x[[index, i]]);
            }
            index += 1;
        }

        result
    }

    fn sigmoid(x: f32) -> f32 {
        1_f32 / (1_f32 + (-x).exp())
    }

    fn sigmoid_derivative(x: f32) -> f32 {
        x * (1_f32 - x)
    }
}
