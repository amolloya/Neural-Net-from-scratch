# Neural-Net-from-scratch

This section shows the implementation of Neural Network from scratch in python without the use of any high-level machine learning libraries, to get a better understanding of how the neural net works from the foundation.

All the steps required for building a fully connected neural network including functions for forward propogation, backward propogation, batch gradient descent, regularization as well as training and testing the data. Using this model we also have the option to tune the hyper-parameters i.e. learning rate, number of epochs, batch size, regularization parameter, etc. to get better accuracies.

We consider the following three classic examples with our neural net developed:
* Iris dataset: For classification of species of Iris (Iris setosa, Iris virginica and Iris versicolor) <br/> Code: NeuralNet-Iris.py <br/> Dataset available at: https://archive.ics.uci.edu/ml/datasets/iris

* MNIST dataset: For classification of numbers (0-9) <br/> Code: NeuralNet-MNIST.py <br/> Dataset available at: http://yann.lecun.com/exdb/mnist/

* Shapes dataset: For classification of shapes (circle, square, triangle) <br/> Code: NeuralNet-Shapes.py <br/> Dataset available at: https://www.kaggle.com/cactus3/basicshapes/version/1
