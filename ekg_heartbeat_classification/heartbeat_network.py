"""
All code written by Rose Ridder and Holden Bridge
All data was taken from https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm via Kaggle
"""

import pandas as pd
import sys, os
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

scratchfilepath = "../../../../scratch/rridder1/" # edit for your username
train_filename = "mitbih_train.csv"
test_filename = "mitbih_test.csv"
small_train_filename = "first8DataEachType_Training.csv"
small_test_filename = "first8DataEachType_Testing.csv"

def getData(full_file_path):
    dat = pd.read_csv(full_file_path, header=None).as_matrix()
    return dat

def manageData():
    train_data = getData(scratchfilepath+train_filename)[67472:] # include only a set of the data for training because there are too many normal beats
    test_data = getData(scratchfilepath+test_filename)[16499:]

    # parse data and reshape
    training, train_target = train_data[:,:-1], train_data[:,-1]
    testing, test_target = test_data[:,:-1], test_data[:,-1]
    training = training.reshape(training.shape[0],training.shape[1], 1)
    testing = testing.reshape(testing.shape[0],testing.shape[1],1)

    num_categories = len(set(test_target))
    train_target = to_categorical(train_target, num_categories)
    test_target = to_categorical(test_target, num_categories)

    # display data shape information
    print("data shapes")
    print("  training input:", training.shape)
    print("  training output :", train_target.shape)
    print("  testing input :", testing.shape)
    print("  testing output:", test_target.shape)

    return training, train_target, testing, test_target

def buildModel(training, train_target, testing, test_target):

    neural_net = Sequential()

    # build a model
    neural_net.add(Conv1D(32, 5 ,activation="relu",input_shape=(187, 1)))
    neural_net.add(MaxPooling1D(pool_size = 4))
    neural_net.add(Conv1D(32, 5 ,activation="relu"))
    neural_net.add(Flatten())
    neural_net.add(Dense(50, activation='relu'))
    neural_net.add(Dense(50, activation='relu'))
    neural_net.add(Dense(5, activation='softmax'))

    neural_net.summary()

    # compile the model
    neural_net.compile(optimizer="Adagrad", loss="categorical_crossentropy",
                       metrics=['categorical_accuracy'])

    # train the model
    history = neural_net.fit(training, train_target, verbose=1,
                             validation_data=(testing, test_target),
                             epochs=50)

    return neural_net

def testModel(neural_net, input, targets):
    # test the model
    loss, accuracy = neural_net.evaluate(input, targets, verbose=0)
    print("accuracy: {}%".format(accuracy*100))
    return accuracy

def loadModel(file_path):
    neural_net = load_model(file_path)
    return neural_net

# These function was used to identify incorrect predictions within a subset of the data
# for easy visualizations
def manageSmallData():
    # retrieve only a subset of data for
    test_data = getData(small_test_filename)

    # parse data and reshape
    testing, test_target = test_data[:,:-1], test_data[:,-1]
    testing = testing.reshape(testing.shape[0],testing.shape[1],1)


    num_categories = len(set(test_target))
    test_target = to_categorical(test_target, num_categories)

    # display data shape information
    print("data shapes")
    print("  testing input :", testing.shape)
    print("  testing output:", test_target.shape)

    return testing, test_target

def compareSubset(neural_net):  # used for assessing a subset of data for visualization purposes
    testing, test_target = manageSmallData()
    print (testing.shape)
    outputs = neural_net.predict(testing)
    answers = [np.argmax(output) for output in outputs]
    targets = [np.argmax(target) for target in test_target]
    import matplotlib.pyplot as plt

    for i in range(len(answers)):
        if answers[i] != targets[i]:
            print(i, "Network predicted", answers[i], "Target is", targets[i])
            plt.plot(range(len(testing[i])), testing[i])
            plt.show()

def main():
    training, train_target, testing, test_target = manageData()

    try:
        file_path = sys.argv[1]
    except:
        file_path = "StandardModel_50_epochs.h5" # load this file if it exists and/or save any new model with this filename


    if os.path.exists(file_path):
        neural_net = loadModel(file_path)
        print ("Loaded existing model")
        neural_net.summary()
    else:
        print ("Building new model")
        neural_net = buildModel(training, train_target, testing, test_target)
        neural_net.save(file_path)

    print ("training accuracy")
    testModel(neural_net, training, train_target)
    print ("testing accuracy")
    testModel(neural_net, testing, test_target)

    print ("done!")

    # compareSubset(neural_net) # used for assessing a subset of data for visualization purposes

if __name__ == "__main__":
    main()
