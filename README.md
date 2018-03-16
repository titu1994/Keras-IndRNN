# IndRNN in Keras
Keras implementation of the IndRNN model from the paper [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

# Usage
Usage of IndRNNCells
```python
from ind_rnn import IndRNNCell, RNN

cells = [IndRNNCell(128), IndRNNCell(128)]
ip = Input(...)
x = RNN(cells)(ip)
...

```

Usage of IndRNN layer
```python
from ind_rnn import IndRNN

ip = Input(...)
x = IndRNN(128)(x)

```

# Notes
IndRnn and its associated cell has additional parameters, `recurrent_clip_min` and `recurrent_clip_max` which default to -1.
-1 indicates that they should take their default values of [0, 2 ^ (1 / Timesteps)] as the clipping range for **ReLU** activation. If you change the activation function, do **not** forget to change the clipping ranges as well, or the model may diverge during training.

In Keras, there is implicit detection of the number of timesteps if the shape is well specified during training. Since this clipping is most important during training (for initialization of weights) and also during inference (for clipping range of the recurrent weights), it is adviseable to **always** specify the number of time steps, even during inference. 

Since it may not be possible to determine the number of timesteps for variable timestep problems, the model defaults to a max clipping range of 1.0, which is equivalent to an infinite timestep problem. This may cause issues if the model was trained using a pre-set timestep.

# Libraries
- Keras 2.1.5+
- Tensorflow / Theano / CNTK (not tested)

# Todo
- [x] Implement IndRNNCell and IndRNN Layer
- [x] Implement IMDB trial
- [x] Implement Addition problem trial
- [x] Implement Sequential MNIST trial
- [x] See if MNIST converges to the paper results. I won't be able to do this since it takes too long to train a 2 layer model on MNIST.

