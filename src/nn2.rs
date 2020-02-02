extern crate ndarray;

use ndarray::Array2;
use rand::prelude::*;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_new_network2() {
        let x = arr2(&[
            [0_f64, 0_f64, 1_f64],
            [0_f64, 1_f64, 1_f64],
            [1_f64, 0_f64, 1_f64],
            [1_f64, 1_f64, 1_f64],
        ]);

        let y = arr2(&[[0_f64], [1_f64], [1_f64], [0_f64]]);
        let mut result = Network2::new(x, y);

        assert_eq!(result.output.shape()[0], 4);

        // test feed_forward_once
        result.feed_forward();

        dbg!(result.loss);
        assert_eq!(1, 0);

        // see if feed forward is working
        /*
        assert_ne!(result.layer1[[0, 0]], 0_f64);

        let before_propagate = result.weights2[[0, 0]];
        result.back_propagation();
        assert_ne!(result.weights2[[0, 0]], before_propagate);
        */
    }
}

// two layer neural network implementation
// extend to run arbitrary number of layers
// also increase to f64
pub struct Network2<'a> {
    input: Array2<f64>,
    y: Array2<f64>,
    output: Array2<f64>,

    // number of layers
    num_layers: u8,
    dimensions: Vec<usize>,
    param: HashMap<&'a str, Array2<f64>>,
    cache: HashMap<&'a str, Array2<f64>>,

    learning_rate: f64,
    sample_size: u8,
    loss: Array2<f64>,
}

// inspired by https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
impl Network2<'_> {
    fn new<'a>(x: Array2<f64>, y: Array2<f64>) -> Network2<'a> {
        let mut rng = rand::thread_rng();
        let mut dimensions = Vec::new();
        // number of neurons
        dimensions.push(15);
        // initialize dimensions, shape of input
        dimensions.push(x.shape()[1]);
        // output layer and second layer of neurons
        dimensions.push(1);

        let weights1 = Array2::<f64>::zeros((dimensions[1], dimensions[0]))
            .mapv(|_| rng.gen_range(0_f64, 1_f64));
        let bias1 = Array2::<f64>::zeros((x.shape()[0], dimensions[0]));
        let weights2 = Array2::<f64>::zeros((dimensions[0], dimensions[2]))
            .mapv(|_| rng.gen_range(0_f64, 1_f64));
        let bias2 = Array2::<f64>::zeros((x.shape()[0], dimensions[2]));

        let mut param = HashMap::new();
        param.insert("W1", weights1);
        param.insert("B1", bias1);
        param.insert("W2", weights2);
        param.insert("B2", bias2);

        let y_shape = y.shape().to_vec();
        Network2 {
            input: x,
            y,
            output: Array2::zeros((y_shape[0], y_shape[1])),
            param,
            num_layers: 2,
            dimensions,
            cache: HashMap::new(),

            learning_rate: 0_f64,
            sample_size: 3,
            loss: Array2::<f64>::zeros((y_shape[1], 1)),
        }
    }

    //Relu and Sigmoid functions activation
    fn feed_forward(&mut self) {
        let z1 = self.input.dot(&self.param["W1"]) + &self.param["B1"];
        let a1 = z1.mapv(Network2::relu);

        let z2 = a1.dot(&self.param["W2"]) + &self.param["B2"];
        let a2 = z2.mapv(Network2::sigmoid);

        self.output = a2.to_owned();
        // cache Z1 for back propagation
        self.cache.insert("Z1", z1);
        self.cache.insert("A1", a1);
        self.cache.insert("Z2", z2);
        self.cache.insert("A2", a2);

        self.loss = self.nloss();
    }

    fn nloss(&mut self) -> Array2<f64> {
        return &self.y - &self.output;
    }

    #[allow(dead_code)]
    fn sigmoid(x: f64) -> f64 {
        1_f64 / (1_f64 + (-x).exp())
    }

    fn relu(x: f64) -> f64 {
        return x.max(0_f64);
    }

    #[allow(dead_code)]
    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1_f64 - x)
    }
}
