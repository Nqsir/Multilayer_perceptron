# Multilayer_perceptron

A fully connected multilayer perceptron using Feedforward, Backward propagation, Gradient descent, Binary Cross Entropy, Softmax, Hyperbolic tangeant, and Mean squared error.

The aim of this project is to assume a prediction using deep learning to figure out if a breast cancer tumor is malignant or benign, based on 30 features (thickness, viscosity, size...).

## Usage:

```py multilayer_perceptron.py data.csv```

## Options:
### Mandatory:
```-t data.csv``` Trainning mode
or
```-p data.csv``` Predict mode

### Optionnal:
```-e``` Defining the number of epochs
```-r``` Defining learning rate
```-s``` Saving a seed
```-l``` Loading a seed

#### Comments on 2020/01/30
The best combinaison of loss vs. time used for prediction is using the seed <seed-clone-9473-20200130.save> with the actual parameters: <learning rate = 0.000045> and <number of epochs = 300>
  
### Training:

![mlp1_ok](https://user-images.githubusercontent.com/40288838/74129582-40320980-4be0-11ea-9459-89cd8435d4c5.png)

### Evaluating the model

![mlp3](https://user-images.githubusercontent.com/40288838/74128599-d31d7480-4bdd-11ea-9b05-f51f23d32d1a.png)

92.98% of predictions are correct, measured with MSE but with 23.09% error using binary cross entropy. The MSE only measures the error rate based on YES or NO the prediction was correct. The Binary cross entropy also measures how accurate was the prediction. Vulgarisation by the example is: if we make the prediction that a tumor is malignant at a 75% rate and the tumor is, the MSE will assume we have a 100% accuracy rate, the binary cross entropy will consider we have 25% error rate.

![mlp2_ok](https://user-images.githubusercontent.com/40288838/74127470-090d2980-4bdb-11ea-8b0c-21e2cfb98f3c.png)
