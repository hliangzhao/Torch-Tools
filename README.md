## Torch Tools
This repos concludes commonly used deep learning models. This repos is based on the book 
https://tangshusen.me/Dive-into-DL-PyTorch.

Module ``linear``:
* ``linear.regression``: implement the linear regression (with and without ``nn.Module``)
* ``linear.softmax``: implement the softmax classification(with and without ``nn.Module``)
* ``linear.svm``: *work-in-progress*
* ``linear.testcase``: implement a case study, where linear regression is used to predict the house price.

Module ``nn``:
* ``nn.mlp``: implement the Multi-Layer Perceptron (with and without ``nn.Module``)
* ``nn.cnn``: implement popular CNNs such as LeNet-5, AlexNet, VGG-11, GooLeNet, etc.
* ``nn.rnn``: *work-in-progress*
* ``nn.model_construct``: concludes typical methods for deep model's construction

Module ``reinforce``: planning to implement value-based and policy-based RL algorithms, such as 
Q-learning, DQN, SARSA, REINFORCE, etc.

Module ``eval``: tests under-fitting, over-fitting, and regularization by regression.

Module ``metrics`` implement typical operations in deep learning, such as the calculation of 
MSE, cross-entropy error, mini-batch gradient descent, drop out, correlation computation, etc.

Module ``tools`` defines utilities. For example, get data in batch, function plot, etc.
