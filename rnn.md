## Recurrent Neural Networks

https://en.wikipedia.org/wiki/Recurrent_neural_network

Recurrent Neural Network (RNN) - are used for input sequences such as text, audio, or video. RNNs are similar to a vanilla neural network but they also pass the hidden state as output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. RNNs are trained by backpropagation through time. This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. Exploding gradient can be resolved by gradient clipping. A common hyperparameter is the number of steps to go back or "unroll" during training. There are variations such as bi-directional and recursive RNNs. Code an RNN
