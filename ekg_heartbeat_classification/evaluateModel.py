"""
All code written by Rose Ridder and Holden Bridge
All data was taken from https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm via Kaggle
"""

from keras.models import load_model
from random import shuffle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

import heartbeat_network as our_functions

def plotMatrix(outputs, target_array):
    mat_train = np.zeros((5,5))

    for irow in range(len(outputs)):
        target = np.argmax(target_array[irow])
        prediction = np.argmax(outputs[irow])
        row = np.zeros_like(outputs[irow])
        row[prediction] = 1

        mat_train[target, prediction] +=1
    print (mat_train)

    fig, ax = plt.subplots()

    ax.matshow(mat_train, cmap=plt.cm.Blues)

    for i in range(mat_train.shape[0]):
        for j in range(mat_train.shape[1]):
            c = mat_train[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.xlabel("Predicted")
    plt.ylabel("Target")

training, train_target, testing, test_target = our_functions.manageData()
file_path = sys.argv[1]
neural_net = our_functions.loadModel(file_path)
print ("Loaded existing model")
neural_net.summary()

train_acc = our_functions.testModel(neural_net, training, train_target)
test_acc = our_functions.testModel(neural_net, testing, test_target)

train_outputs = neural_net.predict(training)
test_outputs = neural_net.predict(testing)

plotMatrix(train_outputs, train_target)
plt.title("Training Data with Accuracy = "+str(train_acc))
plotMatrix(test_outputs, test_target)
plt.title("Testing Data with Accuracy  = "+str(test_acc))
plt.show()
