# Deep Learning

## Neural Networks

- **neural networks**
  - perform well on prediction problems
  - account for **interactions**
      - instead of variables factoring into a single function, can use intermediary function to account for specific relationship between a subset of variables, and feed that function to another downstream function that takes into account (in/directly) all variables
      - can have multiple interactions between the same variables
  - can visualize this as a directed graph
      - leaves / far left nodes of graph are the **input layer** aka **predictive variables**
      - parent node / far right node of graph is the **output layer** aka **target variable**
      - everything in between is a **hidden layer**
        - the more nodes in a hidden layer, the more interactions being captured
      - edges (aka **synapses**) between nodes (aka **neurons**) in **input layer** and **hidden layer** are **weighted**
          - weights are parameters that are determined by training/fitting to data
  - **forward propagation** is process of feeding values for input layer variables, through hidden layer, and outputting a target variable value
      - assuming linear relationships:
        - for a hidden layer node n with input edges (e1, e2, ..., ek) each with weight (w1, w2, ..., wk), sum(for i in (0..k): ei * w1)
            - aka **dot product**
        - the outputted value is the value at node n, and is used to feed values into the output layer
      - to account for non-linearities, instead of just calculating a dot product, each node has an **activation function**
  - **activation function** is applied to node inputs to produce node output
      - before, `tanh` was a common activation function e.g. `tanh(<result of dot product>)`
      - now, standard is **ReLU (rectified linear activation)**
          ```python
          def relu(x)
            if x < 0: return 0
            return x
          ```
      - above, activation function was **identity function** i.e. node output was same as its input

```python
def predict_with_network(input_data_row, weights):
    # calculate first hidden layer composed of 2 nodes
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # calculate output layer
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    return(model_output)


# calculate, store, and print predictions
results = []
for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))

print(results)
```

## Deep Learning

- difference between historical neural networks, and modern deep neural networks (e.g. as used in robotics, NLP, AI, etc) is addition of many hidden layers v a single one i.e. use of **deep networks** i.e. powerful **neural networks**
- deep networks internally build _representations_ of patterns in data so aka **representation learning**
    - when fitted to data, learn weights and assign them to patterns
    - can thus detect patterns e.g. early hidden layers might find patterns for diagonal lines, later nodes might use them to recognize a face

- as will supervised learning, measure performance of model via **loss functions** which aggregates **error** across multiple data points into single number
  - **error** e.g. error = predicted - actual
      - regression tasks might use **mean squared error (MSE)**
  - more data points == more potential for error
  - goal is to find **weights** that minimize loss function

- **gradient descent**: method for finding weights that minimize loss function; weights are needed for edges of a neural network
    - consider a topography that is the result of a 3D graph of weight 1, weight 2, and the output of a loss function
    - algorithm:
      - start at random point (i.e. at specific value of w1 and w2)
      - take a step downhill (i.e. detect slope at point [derivative] and move to a (w1, w2) pair that brings loss function output lower)
          - don't want to take too big a step
          - magnitude of step (i.e. amount to change (w1, w2) by) should be `learning_rate * slope`
          - learning rates often around `0.01`
      - repeat above until uphill in every direction
    - finding the slope (??)
      - for **MSE**: `2 * x * (y-xb)` = `2 * input_data * error`

      ```python
      # calculate the slope
      preds = (input_data * weights).sum()
      error = preds - target
      slope = 2 * input_data * error

      # update the weights
      learning_rate = 0.01
      weights_updated = weights - learning_rate * slope

      # recalculate predictions/error
      preds_updated = (weights_updated * input_data).sum()
      error_updated = preds_updated - target
      ```

- **backpropagation** aka backprop or backwards propagation: does the reverse of **forward propagation** i.e. taking error from output layer and propagating it backwards towards the input layer
    - allows gradient descent to update all weights in a neural network
    - must use **forward propagation** first to get predictions/errors
    - _each_ time you generate predictions w/ fowards propagation, udpate weights w/ backwards propagation
    - algorithm:
      - start at random set of weights
      - use forward propagation to make a prediction
      - use backward propagation to get slope of loss function wrt weights
      - multiple slope by learning rate, subtract result from current weights
      - keep going until you find a flat part (??)
    - for computational efficiency, generally run only on a subset of data, a **batch** i.e. use **stochastic gradient descent**
        - use different batches on each update
        - once all data is used (an **epoch** is complete), start over and re-cycle through the data in batches


## Keras

- **Keras**: a cutting edge library for deep learning in Python
  - interface to **tensorflow** deep learning library
- to build a model:
  - specify architecture: # layers, # nodes per layer, activation function
      - use `"relu"` for regression task activation function
      - use **softmax** for classification task activation function (ensures values in `[0, 1]` so they can be interpreted as probabilities)
  - compile the model: specify loss function, optimizer
      - optimizer controls learning rate; for keras, using `"adam"` is common (adjusts learning rate as it does gradient descent)
          - another option is `'sgd'` for stochastic gradient descent
      - MSE is common loss function for regression problems
      - **categorical cross entropy** is a common loss function for classification problems (similar to log loss), lower is better
  - fit the model to data: apply back propagation w/ gradient descent with data to update weights
      - good idea to scale data to all be roughly on the same scale before optimizing (subtract mean and divide by std dev)

#### Example w/ Keras for a Regression Task

```python
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model

# read data
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
# n_cols is # predictors aka # nodes in input layer
n_cols = predictors.shape[1]

# sequential models specify that layers should have connections only to
# immediately preceding/subsequential layer
model = Sequential()

# add layers
# dense layers connect all nodes in previous layer to all nodes in current layer
# Dense(<# nodes in layer>, <activation function>)
# first layer must specify input_shape
model.add(Dense(100, activation='relu', input_shape=(n_cols, )))
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_square_error')

# target is a matrix w/ data to be predicted
model.fit(predictors, target)

# save a model to HDF5 file and reload it again
model.save('model_file.h5')
model_reloaded = load_model('model_file.h5')
prediction = model_reloaded.predict(some_data)
```

#### Example w/ Keras for a Classification Task

```python
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# read data
data = pd.read_csv('basketball_shot_log.csv')
# drop categorical column 'shot_result' that can have values 0 or 1
# get back as np matrix
predictors = data.drop(['shot_result'], axis=1).as_matrix()
# convert single column 'shot_result' to multiple columns, one for each class
target = to_categorical(data['shot_result'])

n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(n_cols, )))
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))

# specify metrics to see how metrics change after each epoch
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(predictors, target)
```

#### Tuning models

- **dying neuron problem**: when a node has weights less than 0 for all synapses
    - ReLU reduces output of nodes with non positive input values to 0 => contributes nothing to model
    - could have model that never outputs 0 (e.g. `tanh` function) but just results in outputs approaching 0

```python
from keras.optimizers import SGD

# create a stochastic gradient descent optimizer with learning rate 0.01
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
```

- validate models with **validation data** i.e. data held out from training data (like test data in supervised learning)
    - uncommon to use k-fold cross validation b/c it's computationally expensive
    - instead use single validation with a large amount of data
    - to specify that 30% of data should be kept for validation: `model.fit(predictors, target, validation_split=0.3)`
        - will output error on validation set i.e. validation score
- **early stopping**: when you stop training when the validation score stops improving


```python
from keras.callbacks import EarlyStopping

# patience = # epochs model can go without improving before we stop training
early_stopping_monitor = EarlyStopping(patience=2)

# by default, keras trains for 10 epochs, can bump # as below
model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])
```

##### To compare models

```python
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
```

- **model capacity**: model's ability to capture predictive patterns in data
    - increasing # layers, increasing # nodes / layer increases model capacity in neural networks
