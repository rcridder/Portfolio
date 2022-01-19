import numpy as np
import scipy.fftpack as spft
import scipy.signal as spsignal
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

from keras.utils import to_categorical

from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import sys, os

filepathmat = "EEG_Data/"#"../../../../scratch/rridder1/EEG_Data/" # edit for your username
filepathcsv = ""

def allFloats(arr): # CSV Function
    for row in arr:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except:
                row[i] = 0#np.nan

def readData(filename, dat): # CSV Function
    df = pd.read_csv(filename)
    fftInd = [i for i,x in enumerate(df.columns) if "fft" in x]
    fourierCol = df[df.columns[fftInd]].as_matrix()
    aInd = [i for i,x in enumerate(df.columns) if "aaa" in x]
    tCol = df[df.columns[aInd]].as_matrix()
    allFloats(tCol)
    allFloats(fourierCol)
    our_fft = spft.fft(tCol, axis = 1)

    if plot:
        for i in [0,1,2,3,4,5]: # plot the first 6 rows
            plt.figure(); plotting(tCol, fourierCol, our_fft, i)
        plt.show()
    # last column is labels
    if dat == "time":
        data = tCol
    elif dat == "freq":
        data = fourierCol
    elif dat == "our_freq":
        data = our_fft
    else: # second command line entry was invalid
        raise Exception('please check your input format and ensure that your second entry is "time", "freq", or "our_freq" (our_freq is beta testing only.)')

    return separateTrainTestTargets(data, df[df.columns[-1]])

def plotting(tCol, fourierCol, our_fft, ind): # CSV Function
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(0, len(tCol[ind,:])), tCol[ind,:])
    plt.title('time series')
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(fourierCol[ind,:])), fourierCol[ind,:])
    plt.title('fft series')
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, len(our_fft[ind,:])), our_fft[ind,:])
    plt.title('our fft series')

def readMatData(filename, fs, num_samps_per_period): # read one matlab data file and filter the data (baseline and 60Hz filter)
    obj = loadmat(filename, struct_as_record = True)
    headers = ['id','tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']
    data = obj['o']['data'][0][0][0:fs*60*30,4:17]
    data = np.swapaxes(data, 0, 1) # time by channel
    intervalSamples = int((fs*60*9)/num_samps_per_period) # how many samples per interval
    intervals = np.arange(fs*30, fs*60*10,intervalSamples) # list of interval indices
    sets = np.empty((3*num_samps_per_period,13,intervalSamples))
    target = []
    for i, segment in enumerate([0, fs*60*10, fs*60*20]):
        for j in range(len(intervals)-1):
            datset = data[:,segment+intervals[j]:segment+intervals[j+1]]
            datset_base_filt = spsignal.filtfilt([1, -1], [1, -.99], datset) # baseline filter and 60-Hz filter
            sets[num_samps_per_period*i+j,:, :] = datset_base_filt
            target.append(i)
    target = np.array(target)
    return sets, target

def getInputsTargets(numfiles, num_samps_per_period, fs):
    if fourier: inputs = np.zeros(shape = (num_samps_per_period*3*numfiles, 13, int(60*fs/4)))
    else: inputs = np.zeros(shape = (num_samps_per_period*3*numfiles, 13, int((fs*60*9)/num_samps_per_period)))
    targets = np.zeros(shape = (num_samps_per_period*3*numfiles))

    for i in range(1,numfiles + 1):
        filename = "eeg_record"+str(i)+".mat"
        try:
            inputi, targetsi = readMatData(filepathmat+filename, fs, num_samps_per_period)
        except: pass
        if fourier:
            for rowind in range(len(inputi)):
                inputi[rowind] = np.abs(spft.fft(inputi[rowind]))
        else: pass
        inputs[i*len(inputi)-len(inputi):i*len(inputi),:,:] = inputi[:,:,:inputs.shape[2]]
        targets [i*len(targetsi)-len(targetsi):i*len(targetsi)] = targetsi
        if plot:
            plt.figure()
            plt.subplot(1,3,1)
            plt.plot(range(inputs.shape[2]), normalizeInputs(inputi[0,0,:inputs.shape[2]]))
            plt.subplot(1,3,2)
            plt.plot(range(inputs.shape[2]), normalizeInputs(inputi[num_samps_per_period,0,:inputs.shape[2]]))
            plt.subplot(1,3,3)
            plt.plot(range(inputs.shape[2]), normalizeInputs(inputi[-1,0,:inputs.shape[2]]))
            plt.show()
            #plt.plot(range(inputs.shape[2]), inputs[i*len(inputi)-len(inputi), 0,:]); plt.show()
    return inputs, targets

def bulkReadMat(numfiles=34, num_samps_per_period=6, fs = 128):
    if fourier: file_end = 'fourier.npy'
    else: file_end = 'time.npy'
    # check if npy file exists already for quicker file reading
    if os.path.exists('numpyinputs'+file_end) and os.path.exists('numpytargets'+file_end):
        inputs = np.load('numpyinputs'+file_end)
        targets = np.load('numpytargets'+file_end)
    else: # file does not exist... read each file individually
        inputs, targets = getInputsTargets(numfiles, num_samps_per_period, fs)
        np.save('numpyinputs'+file_end, inputs)
        np.save('numpytargets'+file_end, targets)

    return separateTrainTestTargets(inputs, targets)

def normalizeInputs(inputs): # get all inputs on a [-1,1] interval
    return inputs/np.max(np.abs(inputs)+.01, axis = 0)

def separateTrainTestTargets(inputs, targets, percent = .8): # separate training and testing data (default 80:20 ratio)
    inputs = normalizeInputs(inputs)
    train_indxs = np.random.choice(len(inputs), int(len(inputs)*percent), replace = False)
    mask = np.zeros((int(len(targets))), dtype = bool)
    mask[train_indxs] = True

    train = inputs[mask]
    test = inputs[np.logical_not(mask)]
    train_targets = targets[mask]
    test_targets = targets[np.logical_not(mask)]
    print (train.shape, train_targets.shape, test.shape, test_targets.shape)

    num_categories = len(np.unique(train_targets))
    train_targets = to_categorical(train_targets, num_categories)
    test_targets = to_categorical(test_targets, num_categories)

    return train, train_targets, test, test_targets

def buildModel(training, train_target, testing, test_target): # build the matlab model
    neural_net = Sequential()

    neural_net.add(Conv1D(32, 5, activation="relu",input_shape=(training.shape[1], training.shape[2])))
    neural_net.add(Dropout(.3))
    neural_net.add(Flatten())
    neural_net.add(Dense(3, activation='softmax'))
    neural_net.summary()

    return neural_net

def mainMat():
    fs = 128
    numfiles = 34
    num_samps_per_period = 18#6
    training, train_target, testing, test_target = bulkReadMat(numfiles, num_samps_per_period, fs)

    neural_net = buildModel(training, train_target, testing, test_target)

    # Compile the model
    neural_net.compile(optimizer="Nadam", loss="categorical_crossentropy",
                       metrics=['categorical_accuracy'])

    # Train the model
    history = neural_net.fit(training, train_target, verbose=1, validation_data=(testing, test_target),
                             epochs=1)

    loss, accuracy = neural_net.evaluate(testing, test_target, verbose=0)
    print("accuracy: {}%".format(accuracy*100))

    train_outputs = neural_net.predict(training)
    test_outputs = neural_net.predict(testing)

    plotMatrix(train_outputs, train_target)
    plt.title("Confusion Matrix for Training Data")
    plotMatrix(test_outputs, test_target)
    plt.title("Confusion Matrix for Testing Data")
    plt.show()

def mainCSV(): # CSV Function
    training, train_target, testing, test_target = readData(filepathcsv+'mental-state.csv', sys.argv[2])

    neural_net = Sequential()
    neural_net.add(Dense(3, activation= 'softmax', input_shape=(len(testing[0]), )))
    neural_net.summary()

    # Compile the model
    neural_net.compile(optimizer="Nadam", loss="categorical_crossentropy",
                       metrics=['categorical_accuracy'])

    # Train the model
    history = neural_net.fit(training, train_target, verbose=1, validation_data=(testing, test_target),
                             epochs=12)

    loss, accuracy = neural_net.evaluate(testing, test_target, verbose=0)
    print("accuracy: {}%".format(accuracy*100))

    # train_outputs = neural_net.predict(training)
    # test_outputs = neural_net.predict(testing)
    #
    # plotMatrix(train_outputs, train_target)
    # plt.title("Confusion Matrix for Training Data")
    # plotMatrix(test_outputs, test_target)
    # plt.title("Confusion Matrix for Testing Data")
    # plt.show()

def plotMatrix(outputs, target_array):
    # a plot to show the neural net prediction vs. the target of each classification
    mat_train = np.zeros((3,3))

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

plot = False # only works if no npy data file exists. Will plot the time or frequency data as specified
if sys.argv[2] == 'time':
    fourier = False
else:
    fourier = True
if sys.argv[1] == 'csv':
    mainCSV()
else:
    mainMat()
