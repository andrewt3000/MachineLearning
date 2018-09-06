## Recurrent Neural Networks
[Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) - RNNs are an extension of neural networks that also pass the hidden state as output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. RNNs are trained by backpropagation through time. RNNs are used for input sequences such as text, audio, or video.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/rnn.png" />  

This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. Exploding gradient can be mitigated by gradient clipping. A common hyperparameter is the number of steps to go back or "unroll" during training. 

There are variations such as bi-directional and recursive RNNs  

### GRU
GRU - Gated Recurrent Unit - Introduced by Cho. Another RNN variant similar but simpler than LSTM. It contains one update gate and combines the hidden state and memory cells among other differences.  

### LSTM
LSTM - [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) - A specialized RNN that is capable of long term dependencies and mitigates the vanishing gradient problem.  

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Blog post by Chris Olah.  

