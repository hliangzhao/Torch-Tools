## Torch Tools
This repos concludes commonly used deep learning models. The implementation of *basic models* are based on the book 
https://tangshusen.me/Dive-into-DL-PyTorch.

Package ``linear``:
* ``linear.regression``: implement the linear regression (with and without ``torch.nn.Module``)
* ``linear.softmax``: implement the softmax classification (with and without ``torch.nn.Module``)
* ``linear.svm``: implement the hinge loss and SVM classifier with ``torch.nn.Module``

Package ``nn``:
* ``nn.mlp``: implement the Multi-Layer Perceptron (with and without ``torch.nn.Module``)
* ``nn.cnn``: implement popular CNNs such as LeNet-5, AlexNet, VGG-11, GoogLeNet, ResNet-18, DenseNet, etc.
* ``nn.rnn``: implement a basic RNN (one-hidden-layer MLP with hidden state)
* ``nn.model_construct``: concludes typical methods for deep model's construction

Package ``reinforce``: planning to implement value-based and policy-based RL algorithms, such as 
Q-learning, DQN, SARSA, REINFORCE, etc.

Package ``eval``: tests under-fitting, over-fitting, and regularization by regression.

Module ``metrics`` implement typical operations in deep learning, such as the calculation of 
MSE, cross-entropy error, mini-batch gradient descent, drop out, correlation computation, etc.

Module ``tools`` defines utilities. For example, get data in batch, function plot, etc.

Package ``examples`` demonstrates how to use the ``Torch-Tools``. 
* **House Price Prediction (MLP)**: implement a linear regression model to predict the house price.
* **Job Shop Scheduling (DQN)**: implement a MLP with dropout layer and a model defined in figs/JSSP_model.png 
to learn the actions.
* **Offline Service Scheduling (DQN)**: *work-in-progress*
* **Playing Video Games (DQN)**: *work-in-progress*
