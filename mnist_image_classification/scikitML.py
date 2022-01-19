import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from mnist_demo import read_mnist
import numpy as np
from time import time


# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
IMAGE_SIZE = 28
iterations = 500 # max number of iterations for the MLP to adjust within (we tried, 10, 50, 100, and 500)
#throws warning if equilibrium is not attained within the specified number of iterations
#warning thrown for at least one test in each max_iter value except 500

imagesTest, labelsTest = read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                'MNIST_data/t10k-labels-idx1-ubyte.gz')
imagesTrain, labelsTrain = read_mnist('MNIST_data/train-images-idx3-ubyte.gz',
                                'MNIST_data/train-labels-idx1-ubyte.gz')
imagesTest = imagesTest.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
imagesTest = imagesTest * (2.0/255.0) - 1.0
imagesTrain = imagesTrain.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
imagesTrain = imagesTrain * (2.0/255.0) - 1.0

X_train, X_test = imagesTrain, imagesTest
Y_train, Y_test = labelsTrain, labelsTest

def main():
    print 'in main. data aqcuired'
    
    #single layer (no hidden-layer)
    print ('slp_adam')
    mlp((), 'adam', 'relu') #default activation = relu

    print ('slp_sgd')
    mlp((), 'sgd', 'relu')

    print ('slp_lbfgs')
    mlp((), 'lbfgs', 'relu')

    #one hidden layer
    print ('adam_one_layer')
    mlp((20), 'adam', 'relu')

    print ('sgd_one_layer')
    mlp((20), 'sgd', 'relu')

    print ('lbfgs_one_layer')
    mlp((20), 'lbfgs', 'relu')

    print ('mlp_20_20_adam')
    mlp((20,20), 'adam', 'relu')

    print ('mlp_20_20_logistic')
    mlp((20,20), 'adam', 'logistic')

    print ('mlp_20_20_identity')
    mlp((20,20), 'adam', 'identity')

    print ('mlp_20_20_tanh')
    mlp((20,20), 'adam', 'tanh')
    
    print ('mlp_50_50_tanh')
    mlp((50,50), 'adam', 'tanh')
    
    print ('mlp_50_20_tanh')
    mlp((50,20), 'adam', 'tanh')
    
    print ('mlp_20_50_tanh')
    mlp((20,50), 'adam', 'tanh')

    print ('mlp_50_50_50_tanh')
    mlp((50,50,50), 'adam', 'tanh')

    print ('mlp_50_50_50_50_50_tanh')
    mlp((50, 50,50,50,50), 'adam', 'tanh')

    print ('mlp_50_20_50_tanh')
    mlp((50,20,50), 'adam', 'tanh')

    print ('mlp_20_50_20_tanh')
    mlp((20,50,20), 'adam', 'tanh')

def mlp(layer_sizes, solve, act):
    mlpModel = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=iterations, alpha=1e-4,
                        solver=solve, activation = act, verbose=False, tol=1e-4, random_state=1)
    t1=time()
    mlpModel.fit(X_train, Y_train)
    t2=time()
    print("Training set score: %f" % mlpModel.score(X_train, Y_train))
    t3=time()
    print("Test set score: %f" % mlpModel.score(X_test, Y_test))
    t4 = time()
    print ('fit: %f' %(t2-t1))
    print ('score training data: %f'%(t3-t2))
    print ('score test data: %f' %(t4-t3))


if __name__=='__main__':
    main()
